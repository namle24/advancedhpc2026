import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

import sys, time
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda, float32 as n_float32

@cuda.jit
def rgb_to_gray_gpu(src, dst):
    x = cuda.threadIdx.x  + cuda.blockIdx.x *   cuda.blockDim.x
    y = cuda.threadIdx.y +  cuda.blockIdx.y *  cuda.blockDim.y
    if x < src.shape[1] and y < src.shape[0]:
        g = (int(src[y, x, 0]) + int(src[y, x, 1]) + int(src[y, x, 2])) // 3
        dst[y, x] = np.uint8(g)

@cuda.jit
def reduce_min_gpu(src, dst):
    cache = cuda.shared.array(256, n_float32)
    localtid = cuda.threadIdx.x
    tid = cuda.threadIdx.x +  cuda.blockIdx.x *   cuda.blockDim.x * 2
    
    val1 = float(src[tid]) if   tid < src.size else 255.0
    val2 = float(src[tid + cuda.blockDim.x]) if (tid + cuda.blockDim.x) < src.size else 255.0
    cache[localtid] = min(val1, val2)
    cuda.syncthreads()

    s = int(cuda.blockDim.x / 2)
    while s > 0:
        if localtid < s:
            cache[localtid] = min(cache[localtid], cache[localtid + s])
        cuda.syncthreads()
        s = s // 2

    if localtid == 0:
        dst[cuda.blockIdx.x] = cache[0]

@cuda.jit
def reduce_max_gpu(src, dst):
    cache = cuda.shared.array(256, n_float32)
    localtid = cuda.threadIdx.x
    tid = cuda.threadIdx.x   + cuda.blockIdx.x *  cuda.blockDim.x * 2


    val1 = float(src[tid]) if tid < src.size else 0.0
    val2 = float(src[tid +   cuda.blockDim.x]) if (tid + cuda.blockDim.x)  < src.size else 0.0
    cache[localtid] = max(val1, val2)
    cuda.syncthreads()

    s = int(cuda.blockDim.x / 2)
    while s > 0:
        if localtid < s:
            cache[localtid] =  max(cache[localtid], cache[localtid + s])
        cuda.syncthreads()
        s = s // 2

    if localtid == 0:
        dst[cuda.blockIdx.x] = cache[0]

@cuda.jit
def stretch_gpu(gray_img, dst, min_val, max_val):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if x < gray_img.shape[1] and y < gray_img.shape[0]:
        g = float(gray_img[y, x])
        if max_val > min_val:
            new_g = ((g - min_val)  / (max_val - min_val))   * 255.0
        else:
            new_g = g

        if new_g > 255.0:
            new_g = 255.0
        elif new_g < 0.0:
            new_g = 0.0

        val = np.uint8(new_g)
        dst[y, x, 0] = dst[y, x, 1] = dst[y, x, 2] = val

def main():
    img = plt.imread('input.jpg')
    h, w, _ = img.shape
    total_pixels = h * w

    devSrc = cuda.to_device(img)
    devGray = cuda.device_array((h, w), np.uint8)
    devDst = cuda.device_array((h, w, 3), np.uint8)

    blockSizes = [(8, 8), (16, 16), (32, 32)]
    gpu_times = []

    for block in blockSizes:
        bx, by = block
        grid = ((w + bx - 1)  // bx, (h + by - 1) // by)

        t0 = time.time()

        rgb_to_gray_gpu[grid, block](devSrc, devGray)
        cuda.synchronize()
        
#        hostGray = devGray.copy_to_host()
#        min_val = float(np.min(hostGray))
#        max_val = float(np.max(hostGray))


        devFlat = devGray.reshape(total_pixels)
        threads_per_block = 256
        blocks_per_grid = (total_pixels +   threads_per_block * 2 - 1) // (threads_per_block * 2)
        devMinOut = cuda.device_array(blocks_per_grid, np.float32)
        devMaxOut = cuda.device_array(blocks_per_grid, np.float32)


        reduce_min_gpu[blocks_per_grid, threads_per_block](devFlat, devMinOut)
        reduce_max_gpu[blocks_per_grid, threads_per_block](devFlat, devMaxOut)
        cuda.synchronize()



        min_val = float(np.min(devMinOut.copy_to_host()))
        max_val = float(np.max(devMaxOut.copy_to_host()))


        stretch_gpu[grid, block](devGray, devDst, min_val, max_val)
        cuda.synchronize()

        gpu_times.append(time.time() - t0)

    plt.imsave('stretched.jpg', devDst.copy_to_host())

    labels = ['8x8', '16x16', '32x32']
    plt.plot(labels, gpu_times, 'o-')
    plt.xlabel('2D Block Size')
    plt.ylabel('Time (s)')
    plt.savefig('reduce_block_size_vs_time.png')

if __name__ == '__main__':
    main()

