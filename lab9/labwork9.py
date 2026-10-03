import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

import sys, time
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda

@cuda.jit
def rgb_to_gray_gpu(src, dst):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if x < src.shape[1] and y < src.shape[0]:
        g = (int(src[y, x, 0]) + int(src[y, x, 1])  +   int(src[y, x, 2])) // 3
        dst[y, x] = np.uint8(g)



@cuda.jit
def apply_equalization_gpu(gray_img, dst, lut):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if x < gray_img.shape[1] and y < gray_img.shape[0]:
        val = gray_img[y, x]
        new_val = lut[val]
        dst[y, x, 0] = dst[y, x, 1] = dst[y, x, 2] = new_val

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
        grid = ((w + bx - 1) // bx, (h + by - 1) // by)

        t0 = time.time()

        rgb_to_gray_gpu[grid, block](devSrc, devGray)
        cuda.synchronize()

        host_gray = devGray.copy_to_host()
        histo = np.zeros(256, dtype=np.int32)
        for y in range(h):
            for x in range(w):
                histo[host_gray[y, x]] += 1

        p = histo.astype(np.float32) / float(total_pixels)
        cdf = np.zeros(256, dtype=np.float32)
        cdf[0] = p[0]
        for i in range(1, 256):
            cdf[i] = cdf[i - 1] + p[i]


        lut = np.uint8(np.clip(cdf * 255.0, 0, 255))
        devLut = cuda.to_device(lut)

        apply_equalization_gpu[grid, block](devGray, devDst, devLut)
        cuda.synchronize()

        gpu_times.append(time.time() - t0)

    plt.imsave('equalized.jpg', devDst.copy_to_host())

    plt.figure()
    plt.bar(range(256), histo, color='black', width=1.0)
    plt.title('Grayscale Histogram')
    plt.xlabel('Intensity')
    plt.ylabel('Pixel Count')
    plt.savefig('histogram.png')

    # Ve bieu do Block Size vs Time (Slide 21)
    plt.figure()
    labels = ['8x8', '16x16', '32x32']
    plt.plot(labels, gpu_times, 'o-')
    plt.xlabel('2D Block Size')
    plt.ylabel('Time (s)')
    plt.savefig('gather_block_size_vs_time.png')

if __name__ == '__main__':
    main()
