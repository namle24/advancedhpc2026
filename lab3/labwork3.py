import time
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda

@cuda.jit
def grayscale(src, dst):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if tidx < src.shape[0]:
        g = np.uint8((int(src[tidx, 0]) + int(src[tidx, 1]) + int(src[tidx, 2])) / 3)
        dst[tidx, 0] = dst[tidx, 1] = dst[tidx, 2] = g

def main():
    img = plt.imread('input.jpg')
    h, w, _ = img.shape
    pixelCount = h * w
    flatSrc = img.reshape(pixelCount, 3)
    t0 = time.time()
    dst_cpu = np.zeros_like(flatSrc)
    for i in range(pixelCount):
        g = np.uint8((int(flatSrc[i, 0]) + int(flatSrc[i, 1]) + int(flatSrc[i, 2])) / 3)
        dst_cpu[i, 0] = dst_cpu[i, 1] = dst_cpu[i, 2] = g
    t_cpu = time.time() - t0
    print("CPU time:", t_cpu)
    plt.imsave('cpu_gray_extra.jpg', dst_cpu.reshape(h, w, 3))
    devSrc = cuda.to_device(flatSrc)
    devDst = cuda.device_array((pixelCount, 3), np.uint8)
    blockSizes = [32, 64, 128, 256, 512, 1024]
    gpuTimes = []
    for blockSize in blockSizes:
        gridSize = (pixelCount + blockSize - 1) // blockSize
        t0 = time.time()
        grayscale[gridSize, blockSize](devSrc, devDst)
        cuda.synchronize()
        t_gpu = time.time() - t0
        gpuTimes.append(t_gpu)
        print("GPU time:", blockSize, t_gpu)
    hostDst = devDst.copy_to_host()
    plt.imsave('gpu_gray_extra.jpg', hostDst.reshape(h, w, 3))
    print("Speedup:", t_cpu / gpuTimes[3])
    plt.plot(blockSizes, gpuTimes, 'o-')
    plt.xlabel('Block Size')
    plt.ylabel('Time (s)')
    plt.savefig('block_size_vs_time.png')

if __name__ == '__main__':
    main()
