import time
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda

@cuda.jit
def grayscale(src, dst):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if i < src.shape[0]:
        gray = 0.299 * src[i, 0] + 0.587 * src[i, 1] + 0.114 * src[i, 2]
        dst[i, 0] = gray
        dst[i, 1] = gray
        dst[i, 2] = gray

def main():
    img = plt.imread('input.jpg')
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)

    h, w, c = img.shape
    pixelCount = h * w
    src = img.reshape(pixelCount, 3)

    t0 = time.time()
    dst_cpu = np.zeros_like(src)
    for i in range(pixelCount):
        gray = 0.299 * src[i, 0] + 0.587 * src[i, 1] + 0.114 * src[i, 2]
        dst_cpu[i, 0] = gray
        dst_cpu[i, 1] = gray
        dst_cpu[i, 2] = gray
    t_cpu = time.time() - t0
    print("CPU time:", t_cpu)
    plt.imsave('cpu_gray.jpg', dst_cpu.reshape(h, w, 3))

    d_src = cuda.to_device(src)
    d_dst = cuda.device_array_like(src)

    blockSizes = [32, 64, 128, 256, 512, 1024]
    gpuTimes = []

    for blockSize in blockSizes:
        gridSize = (pixelCount + blockSize - 1) // blockSize

        t0 = time.time()
        grayscale[gridSize, blockSize](d_src, d_dst)
        cuda.synchronize()
        t_gpu = time.time() - t0

        gpuTimes.append(t_gpu)
        print("GPU time:", blockSize, t_gpu)

    dst_gpu = d_dst.copy_to_host()
    plt.imsave('gpu_gray.jpg', dst_gpu.reshape(h, w, 3))

    print("Speedup:", t_cpu / gpuTimes[3])

    plt.plot(blockSizes, gpuTimes, 'o-')
    plt.xlabel('Block Size')
    plt.ylabel('Time (s)')
    plt.savefig('block_size_vs_time.png')

if __name__ == '__main__':
    main()
