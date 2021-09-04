# http://www.aholme.co.uk/GPU_FFT/Main.htm
# http://www.peteronion.org.uk/FFT/FastFourier.html

# You need to use sudo because of the GPU
# sudo -E python test_gpu_fft2d.py

import ctypes
from math import log2
import numpy as np
import pathlib

gpu_fft2d = ctypes.CDLL(pathlib.Path.cwd().joinpath('gpu_fft_2d.so'))

fft2d = gpu_fft2d.fft2d
fft2d.restype = ctypes.c_int32
fft2d.argtypes = [
    ctypes.c_int32,
    np.ctypeslib.ndpointer(dtype=np.float32, flags='ALIGNED,C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, flags='ALIGNED,C_CONTIGUOUS')]

ifft2d = gpu_fft2d.ifft2d
ifft2d.restype = ctypes.c_int32
ifft2d.argtypes = [
    ctypes.c_int32,
    np.ctypeslib.ndpointer(dtype=np.float32, flags='ALIGNED,C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, flags='ALIGNED,C_CONTIGUOUS')]


def check_error(res):
    assert res != -1, "Unable to enable V3D. Please check your firmware is up to date.\n"
    assert res != -2, f"Shape not supported.  Try N between {2**8} and {2**22}.\n"
    assert res != -3, "Out of memory. Try a smaller batch or increase GPU memory.\n"
    assert res != -4, "Unable to map Videocore peripherals into ARM memory space.\n"
    assert res != -5, "Can't open libbcm_host.\n"

def gpu_fft2d(input_array):
    assert len(input_array.shape) == 2
    assert input_array.shape[0] == input_array.shape[1]

    N = input_array.shape[0]
    output_array = np.empty((N,2*N),dtype=np.float32)
    res = fft2d(int(log2(N)), input_array.astype(np.float32), output_array)
    
    if res != 0:
        check_error(res)

    return output_array.view(dtype=np.complex64)

def gpu_ifft2d(input_array):
    assert len(input_array.shape) == 2
    assert input_array.shape[0] == input_array.shape[1]

    N = input_array.shape[0]
    
    input_complex = input_array.astype(np.complex64)

    output_array = np.empty((N,2*N),dtype=np.float32)
    res = ifft2d(int(log2(N)), input_complex.view(dtype=np.float32).reshape((N,2*N)), output_array)
    
    if res != 0:
        check_error(res)

    return output_array.view(dtype=np.complex64)


if __name__ == "__main__":
    """Testing...
    """
    import time
    # import matplotlib.pyplot as plt

    N = 1024
    i = np.ones((N,N), dtype=np.float32)*3.1415

    i[200:300,200:300] = 0.0

    time_init = time.monotonic()
    
    res = gpu_fft2d(i)
    i2 = gpu_ifft2d(res)
    print(f"GPU FFT/IFFT time {N}x{N}: {time.monotonic()-time_init}")
    print("FFT2D")
    print(res[0,:4])
    print(res.shape,res.dtype)
    print("IFFT2D")
    print(i2[0,:4])
    print(i2.shape,i2.dtype)

    time_init = time.monotonic()
    res = np.fft.fft2(i)
    i2 = np.fft.ifft2(res)
    print(f"CPU FFT/IFFT time {N}x{N}: {time.monotonic()-time_init}")
    print("FFT2D")
    print(res[0,:4])
    print(res.shape,res.dtype)
    print("IFFT2D")
    print(i2[0,:4])
    print(i2.shape,i2.dtype)

    # # plt.figure(figsize=(10,10))
    # # plt.imshow(re, cmap='gray')
    # # plt.show()
