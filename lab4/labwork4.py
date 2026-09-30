import time
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda

@cuda.jit
def grayscale2D(src, dst):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if x < src.shape[1] and y < src.shape[0]:
        g = np.uint8((int(src[y, x, 0]) + int(src[y, x, 1]) + int(src[y, x, 2])) / 3)
        dst[y, x, 0] = dst[y, x, 1] = dst[y, x, 2] = g

def main():
    img = plt.imread('input.jpg')
    h, w, _ = img.shape
    t0 = time.time()
    dst_cpu = np.zeros_like(img)
    for y in range(h):
        for x in range(w):
            g = np.uint8((int(img[y, x, 0]) + int(img[y, x, 1]) + int(img[y, x, 2])) / 3)
            dst_cpu[y, x, 0] = dst_cpu[y, x, 1] = dst_cpu[y, x, 2] = g
    t_cpu = time.time() - t0
    print("CPU time:", t_cpu)
    plt.imsave('cpu_gray.jpg', dst_cpu)
    devSrc = cuda.to_device(img)
    devDst = cuda.device_array((h, w, 3), np.uint8)
    blockSizes = [(8, 8), (16, 16), (32, 32) ] #max 1024 -> can not (64,64) | dimension must same (16,32), (32,16)
    gpuTimes = []

    for block in blockSizes:
        bx, by = block
        gx = (w + bx - 1) // bx
        gy = (h + by - 1) // by
        grid = (gx, gy)
        t0 = time.time()
        grayscale2D[grid, block](devSrc, devDst)
        cuda.synchronize()
        t_gpu = time.time() - t0
        gpuTimes.append(t_gpu)
        print("GPU time:", block, t_gpu)
    hostDst = devDst.copy_to_host()
    plt.imsave('gpu_gray.jpg', hostDst)
    speedups = [t_cpu / t for t in gpuTimes]
    print("Speedups:", speedups)
    blockLabels = ['8x8', '16x16', '32x32']
    plt.plot(blockLabels, speedups, 'o-')
    plt.xlabel('2D Block Size')
    plt.ylabel('Speedup')
    plt.savefig('block_size_vs_speedup.png')

if __name__ == '__main__':
    main()
