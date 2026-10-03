import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

import sys, time
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda

@cuda.jit
def rgb2hsv_gpu(src, outH, outS, outV):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if x < src.shape[1] and y < src.shape[0]:
        r = float(src[y, x, 0]) / 255.0
        g = float(src[y, x, 1]) / 255.0
        b = float(src[y, x, 2]) / 255.0

        max_c = max(r, max(g, b))
        min_c = min(r, min(g, b))
        df = max_c - min_c

        if df == 0.0:
            h = 0.0
        elif max_c == r:
            h = 60.0 * (((g - b) / df) % 6.0)
        elif max_c == g:
            h = 60.0 * (((b - r) / df) + 2.0)
        elif max_c == b:
            h = 60.0 * (((r - g) / df) + 4.0)

        if max_c == 0.0:
            s = 0.0
        else:
            s = df / max_c

        v = max_c

        outH[y, x] = float(h)
        outS[y, x] = float(s)
        outV[y, x] = float(v)

@cuda.jit(max_registers=64)
def kuwahara_gpu(src, inV, dst, window_size):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    h, w, _ = src.shape

    if x < w and y < h:
        w_size = window_size


        min_var = 1e9
        best_r = 0.0
        best_g = 0.0
        best_b = 0.0

        for k in range(4):
            if k == 0:
                x_min_off, x_max_off, y_min_off, y_max_off = -w_size, 0, -w_size, 0
            elif k == 1:
                x_min_off, x_max_off, y_min_off, y_max_off = 0, w_size, -w_size, 0
            elif k == 2:
                x_min_off, x_max_off, y_min_off, y_max_off = -w_size, 0, 0, w_size
            else:
                x_min_off, x_max_off, y_min_off, y_max_off = 0, w_size, 0, w_size

            sum_v = 0.0
            sum_v2 = 0.0
            sum_r = 0.0
            sum_g = 0.0
            sum_b = 0.0
            count = 0


            for dy in range(y_min_off, y_max_off + 1):
                for dx in range(x_min_off, x_max_off + 1):
                    px = min(max(x + dx, 0), w - 1)
                    py = min(max(y + dy, 0), h - 1)

                    val_v = float(inV[py, px])
                    sum_v += val_v
                    sum_v2 += val_v * val_v

                    sum_r += float(src[py, px, 0])
                    sum_g += float(src[py, px, 1])
                    sum_b += float(src[py, px, 2])
                    count += 1

            mean_v = sum_v / float(count)
            var_v = (sum_v2 / float(count)) - (mean_v * mean_v)

            if var_v < min_var:
                min_var = var_v
                best_r = sum_r / float(count)
                best_g = sum_g / float(count)
                best_b = sum_b / float(count)

        dst[y, x, 0] = np.uint8(best_r)
        dst[y, x, 1] = np.uint8(best_g)
        dst[y, x, 2] = np.uint8(best_b)

def main():
    img = plt.imread('input.jpg')
    h, w, _ = img.shape

    devSrc = cuda.to_device(img)
    devH = cuda.device_array((h, w), np.float32)
    devS = cuda.device_array((h, w), np.float32)
    devV = cuda.device_array((h, w), np.float32)
    devDst = cuda.device_array((h, w, 3), np.uint8)


    win_size = 3

    blockSizes = [(8, 8), (16, 16)]
    gpu_times = []


    for block in blockSizes:
        bx, by = block
        grid = ((w + bx - 1) // bx, (h + by - 1) // by)

        t0 = time.time()

        rgb2hsv_gpu[grid, block](devSrc, devH, devS, devV)
        cuda.synchronize()

        kuwahara_gpu[grid, block](devSrc, devV, devDst, win_size)
        cuda.synchronize()

        gpu_times.append(time.time() - t0)


    plt.imsave('kuwahara.jpg', devDst.copy_to_host())
    labels = ['8x8', '16x16']
    plt.plot(labels, gpu_times, 'o-')
    plt.xlabel('2D Block Size')
    plt.ylabel('Time (s)')
    plt.savefig('kuwahara_block_size_vs_time.png')

if __name__ == '__main__':
    main()
