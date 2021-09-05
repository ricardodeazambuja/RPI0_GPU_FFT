# RPI0_GPU_FFT
Experiments using the RPI Zero GPU for FFT/IFFT 1D/2D

**For an input 4194304 (1D), the GPU was around 7X faster than np.fft.fft and np.fft.ifft in sequence.**

**For an input 1024x1024 (2D), the GPU was around 2X faster than np.fft.fft2 and np.fft.ifft2 in sequence.**

The CPU is always faster for small arrays (and the min size for GPU is 256).

## Instalation
Clone this repo (or simply download all files to your RPI0) and run ```make```. 
It will generate a file ```rpi0_gpu_fft.so``` that you should have in the same directory as the ```rpi0_gpu_fft.py```. That's it ;)

I'm supposing your RPI0 has the rest of the files necessary installed by default (```/opt/vc/src/hello_pi/hello_fft/```).

## TL;DR
Totally untested and without any guarantee what-so-ever :D  

You can try it by running:
```$ sudo -E python rpi0_gpu_fft.py```.

## Original code / ideas from:
* http://www.aholme.co.uk/GPU_FFT/Main.htm
* http://www.peteronion.org.uk/FFT/FastFourier.html
