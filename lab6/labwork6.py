import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

import sys,time
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda

@cuda.jit
def binarization_gpu(src,dst,threshold):
    x = cuda.threadIdx.x   + cuda.blockIdx.x  * cuda.blockDim.x
    y = cuda.threadIdx.y +  cuda.blockIdx.y  * cuda.blockDim.y
    if x < src.shape[1] and y < src.shape[0]:
        g = (int(src[y, x, 0]) +   int(src[y, x, 1])   + int(src[y, x, 2])) // 3
        if g>=threshold:
            val = np.uint8(255)
        else:
            val=np.uint8(0)
        dst[y,x,0] = dst[y,x,1] = dst[y,x,2] = val

@cuda.jit
def brightness_gpu(src, dst, delta):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y+cuda.blockIdx.y*cuda.blockDim.y
    if x<src.shape[1] and y<src.shape[0]:
        for c in range(3):
            val = int(src[y, x, c]) + delta
            if val>255:
                val = 255
            elif val < 0:
                val=0
            dst[y,x,c]=np.uint8(val)

@cuda.jit
def blend_gpu(src1,src2,dst,c):
    x=cuda.threadIdx.x+cuda.blockIdx.x*cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if x < src1.shape[1] and y < src1.shape[0]:
        for ch in range(3):
            val = c * float(src1[y,x,ch]) + (1.0 - c) * float(src2[y,x,ch])
            dst[y,x,ch] = np.uint8(val)

def main():
    img1 = plt.imread('input.jpg')
    h, w, _ = img1.shape

    threshold = int(sys.argv[1]) if len(sys.argv)>1 else 128
    brightness_delta=67
    blend_c = 0.67

    img2 = np.ascontiguousarray(np.flip(img1, axis=1))

    devSrc1 = cuda.to_device(img1)
    devSrc2 = cuda.to_device(img2)
    devDst = cuda.device_array((h,w,3), np.uint8)

    blockSizes = [(8, 8), (16, 16), (32, 32)]
    
    for block in blockSizes:
        bx, by = block
        grid = ((w + bx - 1) // bx, (h + by - 1) // by)
        binarization_gpu[grid, block](devSrc1, devDst, threshold)
        cuda.synchronize()
    plt.imsave('binarization.jpg', devDst.copy_to_host())

    for block in blockSizes:
        bx, by = block
        grid=((w+bx-1)//bx, (h+by-1)//by)
        brightness_gpu[grid, block](devSrc1, devDst, brightness_delta)
        cuda.synchronize()
    plt.imsave('brightness.jpg', devDst.copy_to_host())

    gpu_times = []
    for block in blockSizes:
        bx,by = block
        grid = ((w + bx - 1) // bx, (h + by - 1) // by)
        t0 = time.time()
        blend_gpu[grid, block](devSrc1, devSrc2, devDst, blend_c)
        cuda.synchronize()
        gpu_times.append(time.time() - t0)
    plt.imsave('blend.jpg', devDst.copy_to_host())

    labels = ['8x8', '16x16', '32x32']
    plt.plot(labels, gpu_times, 'o-')
    plt.xlabel('2D Block Size')
    plt.ylabel('Time (s)')
    plt.savefig('map_block_size_vs_time.png')

if __name__ == '__main__':
    main()
