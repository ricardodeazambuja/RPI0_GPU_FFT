# RPI0_GPU_FFT
Experiments using the RPI Zero GPU for FFT/IFFT 1D/2D

**For an input 4194304 (1D), the GPU was around 7X faster than np.fft.fft and np.fft.ifft in sequence.**

**For an input 1024x1024 (2D), the GPU was around 2X faster than np.fft.fft2 and np.fft.ifft2 in sequence.**

The CPU is always for small arrays (and the min size for GPU is 256).

## TL;DR
Totally untested :D

## Original code / ideas from:
* http://www.aholme.co.uk/GPU_FFT/Main.htm
* http://www.peteronion.org.uk/FFT/FastFourier.html
