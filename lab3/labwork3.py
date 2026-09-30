import time
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda

@cuda.jit
def grayscale(src, dst):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if tidx < src.shape[0]:
        g = np.uint8((src[tidx, 0] + src[tidx, 1] + src[tidx, 2]) / 3)
        dst[tidx, 0] = dst[tidx, 1] = dst[tidx, 2] = g

def main():
    img = plt.imread('input.jpg')
    h, w, _ = img.shape
    pixelCount = h * w
    flatSrc = img.reshape(pixelCount, 3)
    dst_cpu = np.zeros_like(flatSrc)
    for i in range(pixelCount):
        g = np.uint8((flatSrc[i, 0] + flatSrc[i, 1] + flatSrc[i, 2]) / 3)
        dst_cpu[i, 0] = dst_cpu[i, 1] = dst_cpu[i, 2] = g
    plt.imsave('cpu_gray_extra.jpg', dst_cpu.reshape(h, w, 3))
    devSrc = cuda.to_device(flatSrc)
    devDst = cuda.device_array((pixelCount, 3), np.uint8)
    blockSize = 64
    gridSize = (pixelCount + blockSize - 1) // blockSize
    grayscale[gridSize, blockSize](devSrc, devDst)
    hostDst = devDst.copy_to_host()
    plt.imsave('gpu_gray_extra.jpg', hostDst.reshape(h, w, 3))

if __name__ == '__main__':
    main()
