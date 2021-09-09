# http://www.aholme.co.uk/GPU_FFT/Main.htm
# http://www.peteronion.org.uk/FFT/FastFourier.html

# You need to use sudo because of the GPU
# sudo -E python rpi0_gpu_fft.py

import ctypes
from math import log2
import numpy as np
import pathlib

gpu_fft = ctypes.CDLL(pathlib.Path.cwd().joinpath('rpi0_gpu_fft.so'))

fft1d = gpu_fft.fft1d
fft1d.restype = ctypes.c_int32
fft1d.argtypes = [
    ctypes.c_uint32,
    ctypes.c_uint32,
    np.ctypeslib.ndpointer(dtype=np.float32, flags='ALIGNED,C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, flags='ALIGNED,C_CONTIGUOUS')]

ifft1d = gpu_fft.ifft1d
ifft1d.restype = ctypes.c_int32
ifft1d.argtypes = [
    ctypes.c_uint32,
    ctypes.c_uint32,
    np.ctypeslib.ndpointer(dtype=np.float32, flags='ALIGNED,C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, flags='ALIGNED,C_CONTIGUOUS')]


fft2d = gpu_fft.fft2d
fft2d.restype = ctypes.c_int32
fft2d.argtypes = [
    ctypes.c_uint32,
    ctypes.c_uint32,
    np.ctypeslib.ndpointer(dtype=np.float32, flags='ALIGNED,C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, flags='ALIGNED,C_CONTIGUOUS')]

ifft2d = gpu_fft.ifft2d
ifft2d.restype = ctypes.c_int32
ifft2d.argtypes = [
    ctypes.c_uint32,
    ctypes.c_uint32,
    np.ctypeslib.ndpointer(dtype=np.float32, flags='ALIGNED,C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, flags='ALIGNED,C_CONTIGUOUS')]


def check_error(res):
    assert res != -1, "Unable to enable V3D. Please check your firmware is up to date.\n"
    assert res != -2, f"Shape {N}x{M} not supported.  Try between {2**8} and {2**22}.\n"
    assert res != -3, "Out of memory. Try a smaller batch or increase GPU memory.\n"
    assert res != -4, "Unable to map Videocore peripherals into ARM memory space.\n"
    assert res != -5, "Can't open libbcm_host.\n"

def gpu_fft1d(input_array):
    assert len(input_array.shape) == 2

    N = input_array.shape[0]
    M = input_array.shape[1]
    assert (M & (M-1) == 0) and M != 0, "Power of 2, please!"

    output_array = np.empty((N,2*M),dtype=np.float32)
    # input_complex = input_array.astype(np.complex64)
    # res = fft1d(N, M, input_complex.view(dtype=np.float32).reshape((N,2*M)), output_array)
    res = fft1d(N, M, input_array.astype(dtype=np.float32), output_array)
    
    if res != 0:
        check_error(res)

    return output_array.view(dtype=np.complex64)

def gpu_ifft1d(input_array):
    assert len(input_array.shape) == 2

    N = input_array.shape[0]
    M = input_array.shape[1]
    assert (M & (M-1) == 0) and M != 0, "Power of 2, please!"

    input_complex = input_array.astype(np.complex64)
    # output_array = np.empty((N,2*M),dtype=np.float32)
    # res = ifft1d(N, M, input_complex.view(dtype=np.float32).reshape((N,2*M)), output_array)
    output_array = np.empty((N,M),dtype=np.float32)
    res = ifft1d(N, M, input_complex.view(dtype=np.float32).reshape((N,2*M)), output_array)
    
    if res != 0:
        check_error(res)

    # return output_array.view(dtype=np.complex64)
    return output_array

def gpu_fft2d(input_array):
    assert len(input_array.shape) == 2

    N = input_array.shape[0]
    assert (N & (N-1) == 0) and N != 0, "Power of 2, please!"
    M = input_array.shape[1]
    assert (M & (M-1) == 0) and M != 0, "Power of 2, please!"
    

    output_array = np.empty((N,2*M),dtype=np.float32)
    res = fft2d(N, M, input_array.astype(np.float32), output_array)
    
    if res != 0:
        check_error(res)

    return output_array.view(dtype=np.complex64)

def gpu_ifft2d(input_array):
    assert len(input_array.shape) == 2

    N = input_array.shape[0]
    assert (N & (N-1) == 0) and N != 0, "Power of 2, please!"
    M = input_array.shape[1]
    assert (M & (M-1) == 0) and M != 0, "Power of 2, please!"


    input_complex = input_array.astype(np.complex64)

    # output_array = np.empty((N,2*M),dtype=np.float32)
    # res = ifft2d(N, M, input_complex.view(dtype=np.float32).reshape((N,2*M)), output_array)
    output_array = np.empty((N,M),dtype=np.float32)
    res = ifft2d(N, M, input_complex.view(dtype=np.float32).reshape((N,2*M)), output_array)

    if res != 0:
        check_error(res)

    # return output_array.view(dtype=np.complex64)
    return output_array


if __name__ == "__main__":
    """Testing...
    """
    import time
    # import matplotlib.pyplot as plt

    N = 1
    M = 2**16
    print(f"Testing the FFT/IFFT 1D... Length {M} for {N} times")
    # i = np.ones((N,M), dtype=np.float32).astype(np.complex64)*3.1415
    i = np.ones((N,M), dtype=np.float32)*3.1415
    i[:,2:int(M/2)] = 0.0
    
    trials = 10
    time_v = []
    for j in range(trials):
        time_init = time.monotonic()
        res = np.fft.fft(i)
        i2 = np.fft.ifft(res)
        time_end = time.monotonic()-time_init
        print(f"CPU FFT/IFFT 1D time {N}x{M}: {time_end}")
        print("FFT1D")
        print(res[0,:4])
        print(res.shape,res.dtype)
        print("IFFT1D")
        print(i2[0,:4])
        print(i2.shape,i2.dtype)
        time_v.append(time_end)
        del res, i2
    
    cpu1_avg = sum(time_v)/trials
    print(f"CPU FFT/IFFT 1D time {N}x{M}: avg time {cpu1_avg}")

    time_v = []
    for j in range(trials):
        time_init = time.monotonic()
        res = gpu_fft1d(i)
        i2 = gpu_ifft1d(res)
        time_end = time.monotonic()-time_init
        print(f"GPU FFT/IFFT 1D time {N}x{M}: {time_end}")
        print("FFT1D")
        print(res[0,:4])
        print(res.shape,res.dtype)
        print("IFFT1D")
        print(i2[0,:4])
        print(i2.shape,i2.dtype)
        time_v.append(time_end)
        del res, i2
    
    gpu1_avg = sum(time_v)/trials
    print(f"GPU FFT/IFFT 1D time {N}x{M}: avg time {gpu1_avg}")
    del i

    print(f"GPU/CPU FFT/IFFT 1D time {N}x{M}: avg time ({trials} trials): {cpu1_avg/gpu1_avg:0.4f}")

    N = 1024
    M = 1024
    print(f"Testing the FFT/IFFT 2D... {M}x{N}")
    i = np.ones((N,M), dtype=np.float32)*3.1415
    i[:,2:int(M/2)] = 0.0
        
    time_v = []
    for j in range(trials):
        time_init = time.monotonic()
        res = np.fft.fft2(i)
        i2 = np.fft.ifft2(res)
        time_end = time.monotonic()-time_init
        print(f"CPU FFT/IFFT 2D time {N}x{M}: {time_end}")
        print("FFT2D")
        print(res[0,:4])
        print(res[-1,:4])
        print(res.shape,res.dtype)
        print("IFFT2D")
        print(i2[0,:4])
        print(i2[-1,:4])
        print(i2.shape,i2.dtype)
        time_v.append(time_end)
        del res, i2
    
    cpu1_avg = sum(time_v)/trials
    print(f"CPU FFT/IFFT 2D time {N}x{M}: avg {cpu1_avg}")
    
    
    time_v = []
    for j in range(trials):
        time_init = time.monotonic()
        res = gpu_fft2d(i)
        i2 = gpu_ifft2d(res)
        time_end = time.monotonic()-time_init
        print(f"GPU FFT/IFFT 2D time {N}x{M}: {time_end}")
        print("FFT2D")
        print(res[0,:4])
        print(res[-1,:4])
        print(res.shape,res.dtype)
        print("IFFT2D")
        print(i2[0,:4])
        print(i2[-1,:4])
        print(i2.shape,i2.dtype)
        time_v.append(time_end)
        del res, i2

    gpu1_avg = sum(time_v)/trials
    print(f"GPU FFT/IFFT 2D time {N}x{M}: avg {gpu1_avg}")

    print(f"GPU/CPU FFT/IFFT 2D time {N}x{M}: avg time ({trials} trials): {cpu1_avg/gpu1_avg:0.4f}")

    # # plt.figure(figsize=(10,10))
    # # plt.imshow(re, cmap='gray')
    # # plt.show()
