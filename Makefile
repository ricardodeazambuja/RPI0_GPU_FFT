D = /opt/vc/src/hello_pi/hello_fft/

S = $(D)hex/shader_256.hex \
    $(D)hex/shader_512.hex \
    $(D)hex/shader_1k.hex \
    $(D)hex/shader_2k.hex \
    $(D)hex/shader_4k.hex \
    $(D)hex/shader_8k.hex \
    $(D)hex/shader_16k.hex \
    $(D)hex/shader_32k.hex \
    $(D)hex/shader_64k.hex \
    $(D)hex/shader_128k.hex \
    $(D)hex/shader_256k.hex \
    $(D)hex/shader_512k.hex \
    $(D)hex/shader_1024k.hex \
    $(D)hex/shader_2048k.hex \
    $(D)hex/shader_4096k.hex

C = $(D)mailbox.c $(D)gpu_fft.c $(D)gpu_fft_base.c $(D)gpu_fft_twiddles.c $(D)gpu_fft_shaders.c $(D)gpu_fft_trans.c

C1D = $(C) rpi0_gpu_fft.c

H1D = $(D)gpu_fft.h $(D)mailbox.h $(D)gpu_fft_trans.h

F = -lrt -lm -ldl

all:	gpu_fft_2d.bin

gpu_fft_2d.bin: $(S) $(C1D) $(H1D)
	gcc -fPIC -shared -o rpi0_gpu_fft.so $(f) $(C1D)
    # gcc -fPIC -shared -ffast-math -mtune=native -O3 -o gpu_fft_2d.so $(f) $(C1D)

clean:
	rm -f *.bin
