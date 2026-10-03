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


@cuda.jit
def hsv2rgb_gpu(inH, inS, inV, dst):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if x < inH.shape[1] and y < inH.shape[0]:
        h = float(inH[y, x])
        s = float(inS[y, x])
        v = float(inV[y, x])

        d = h / 60.0
        hi = int(d) % 6
        f = d - float(int(d))

        l = v * (1.0 - s)
        m = v * (1.0 - f * s)
        n = v * (1.0 - (1.0 - f) * s)

        if hi == 0:
            r, g, b = v, n, l
        elif hi == 1:
            r, g, b = m, v, l
        elif hi == 2:
            r, g, b = l, v, n
        elif hi == 3:
            r, g, b = l, m, v
        elif hi == 4:
            r, g, b = n, l, v
        else:
            r, g, b = v, l, m

        r_out = r * 255.0
        g_out = g * 255.0
        b_out = b * 255.0


        if r_out > 255.0: r_out = 255.0
        if g_out > 255.0: g_out = 255.0
        if b_out > 255.0: b_out = 255.0
        if r_out < 0.0: r_out = 0.0
        if g_out < 0.0: g_out = 0.0
        if b_out < 0.0: b_out = 0.0

        dst[y, x, 0] = np.uint8(r_out)
        dst[y, x, 1] = np.uint8(g_out)
        dst[y, x, 2] = np.uint8(b_out)

def main():
    img = plt.imread('input.jpg')
    h, w, _ = img.shape

    devSrc = cuda.to_device(img)
    devH = cuda.device_array((h, w), np.float32)
    devS = cuda.device_array((h, w), np.float32)
    devV = cuda.device_array((h, w), np.float32)
    devDst = cuda.device_array((h, w, 3), np.uint8)

    blockSizes = [(8, 8), (16, 16), (32, 32)]
    gpu_times = []




    for block in blockSizes:
        bx, by = block
        grid = ((w + bx - 1) // bx, (h + by - 1) // by)

        t0 = time.time()

        rgb2hsv_gpu[grid, block](devSrc, devH, devS, devV)
        cuda.synchronize()

        hsv2rgb_gpu[grid, block](devH, devS, devV, devDst)
        cuda.synchronize()

        gpu_times.append(time.time() - t0)

    plt.imsave('reconstructed.jpg', devDst.copy_to_host())

    labels = ['8x8', '16x16', '32x32']
    plt.plot(labels, gpu_times, 'o-')
    plt.xlabel('2D Block Size')
    plt.ylabel('Time (s)')
    plt.savefig('scatter_block_size_vs_time.png')

if __name__ == '__main__':
    main()
