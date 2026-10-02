import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

import time
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda, float32


FILTER = np.array([
    [0,  0,  1,   2,  1,  0, 0],
    [0,  3, 13,  22, 13,  3, 0],
    [1, 13, 59,  97, 59, 13, 1],
    [2, 22, 97, 159, 97, 22, 2],
    [1, 13, 59,  97, 59, 13, 1],
    [0,  3, 13,  22, 13,  3, 0],
    [0,  0,  1,   2,  1,  0, 0]
], dtype=np.float32)


@cuda.jit
def blur_gpu(src, dst, flt):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    h, w, _ = src.shape

    if x < w and y < h:
        for c in range(3):
            s = 0.0
            for ky in range(-3, 4):
                for kx in range(-3, 4):
                    py = y + ky
                    px = x + kx
                    if py >= 0 and py < h and px >= 0 and px < w:
                        s += src[py, px, c] * flt[ky + 3, kx + 3]
            dst[y, x, c] = np.uint8(s / 1003.0)


@cuda.jit
def blur_shared_gpu(src, dst, flt):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    h, w, _ = src.shape


    s_flt = cuda.shared.array((7, 7), float32)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    if tx < 7 and ty < 7:
        s_flt[ty, tx] = flt[ty, tx]

    cuda.syncthreads()
    if x < w and y < h:
        for c in range(3):
            s = 0.0
            for ky in range(-3, 4):
                for kx in range(-3, 4):
                    py = y + ky
                    px = x + kx
                    if py >= 0  and py < h and  px >= 0 and px < w:
                        s += src[py, px, c] * s_flt[ky + 3, kx + 3]
            dst[y, x, c] = np.uint8(s / 1003.0)

def main():
    img = plt.imread('input.jpg')
    h, w, _ = img.shape

    t0 = time.time()
    dst_cpu = np.zeros_like(img)
    for y in range(h):
        for x in range(w):
            for c in range(3):
                s = 0.0
                for ky in range(-3, 4):
                    for kx in range(-3, 4):
                        py = y + ky
                        px = x + kx
                        if py >= 0  and py < h and  px >= 0 and px < w:
                            s += img[py, px, c] * FILTER[ky + 3, kx + 3]
                dst_cpu[y, x, c] = np.uint8(s / 1003.0)

    t_cpu = time.time() - t0
    print("CPU time:", t_cpu)
    plt.imsave('cpu_blur.jpg', dst_cpu)


    devSrc = cuda.to_device(img)
    devDst1 = cuda.device_array((h, w, 3), np.uint8)
    devDst2 = cuda.device_array((h, w, 3), np.uint8)
    devFlt = cuda.to_device(FILTER)

    blockSizes = [(8, 8), (16, 16), (32, 32)]
    speedup_no_shared = []
    speedup_shared = []


    for block in blockSizes:
        bx, by = block
        gx = (w + bx - 1) // bx
        gy = (h + by - 1) // by
        grid = (gx, gy)

        # GPU khong shared
        t0 = time.time()
        blur_gpu[grid, block](devSrc, devDst1, devFlt)
        cuda.synchronize()
        t_gpu1 = time.time()  - t0
        speedup_no_shared.append(t_cpu / t_gpu1)

        # GPU co shared
        t0 = time.time()
        blur_shared_gpu[grid, block](devSrc, devDst2, devFlt)
        cuda.synchronize()
        t_gpu2 = time.time() -  t0
        speedup_shared.append(t_cpu / t_gpu2)

    plt.imsave('gpu_blur.jpg', devDst2.copy_to_host())
    labels = ['8x8', '16x16', '32x32']
    plt.plot(labels, speedup_no_shared, 'o-', label='No Shared Memory')
    plt.plot(labels, speedup_shared, 's--', label='With Shared Memory')
    plt.xlabel('2D Block Size')
    plt.ylabel('Speedup')
    plt.legend()
    plt.savefig('block_size_vs_speedup_blur.png')

if __name__ == '__main__':
    main()
