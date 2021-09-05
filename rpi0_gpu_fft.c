/*
BCM2835 "RPI_GPU_FFT", modified from:

BCM2835 "GPU_FFT" release 2.0
Copyright (c) 2014, Andrew Holme.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the copyright holder nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <string.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>

#include "gpu_fft_trans.h"
#include "mailbox.h"


#define GPU_FFT_ROW(fft, io, y) ((fft)->io+(fft)->step*(y))


// https://stackoverflow.com/a/11398748/7658422
const int tab32[32] = {
     0,  9,  1, 10, 13, 21,  2, 29,
    11, 14, 16, 18, 22, 25,  3, 30,
     8, 12, 20, 28, 15, 17, 24,  7,
    19, 27, 23,  6, 26,  5,  4, 31};

int log2_32 (uint32_t value)
{
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    return tab32[(uint32_t)(value*0x07C4ACDD) >> 27];
}

int fft1d(uint32_t N, uint32_t M, float *input_array, float *output_array) {
    
    int r, c, ret, mb = mbox_open();

    int log2_N = log2_32(N);
    int log2_M = log2_32(M);

    struct GPU_FFT_COMPLEX *row;
    struct GPU_FFT *fft_pass[1];


    // Prepare 1st FFT pass
    ret = gpu_fft_prepare(mb, log2_M, GPU_FFT_FWD, N, fft_pass);
    if (ret) {
        return ret;
    }

    // Setup input data
    for (r=0;r<N;r++){
        row = GPU_FFT_ROW(fft_pass[0], in, r);
        for (c=0;c<M;c++){
            row[c].re = input_array[(r*M+c)];
            // row[c].re = input_array[2*(r*M+c)];
            // row[c].im = input_array[2*(r*M+c)+1];
        }
    }

    gpu_fft_execute(fft_pass[0]);

    // Write output data
    for (r=0; r<N; r++) {
        row = GPU_FFT_ROW(fft_pass[0], out, r);
        for (c=0; c<M; c++) {
            output_array[2*(r*M+c)] = row[c].re;
            output_array[2*(r*M+c)+1] = row[c].im;
        }
    }

    // Clean-up properly.  Videocore memory lost if not freed !
    gpu_fft_release(fft_pass[0]);

    return 0;
}

int ifft1d(uint32_t N, uint32_t M, float *input_array, float *output_array) {
    
    int r, c, ret, mb = mbox_open();

    int log2_N = log2_32(N);
    int log2_M = log2_32(M);

    struct GPU_FFT_COMPLEX *row;
    struct GPU_FFT *fft_pass[1];


    // Prepare 1st FFT pass
    ret = gpu_fft_prepare(mb, log2_M, GPU_FFT_REV, N, fft_pass);
    if (ret) {
        return ret;
    }

    // Setup input data
    for (r=0;r<N;r++){
        row = GPU_FFT_ROW(fft_pass[0], in, r);
        for (c=0;c<M;c++){
            row[c].re = input_array[2*(r*M+c)]/M;
            row[c].im = input_array[2*(r*M+c)+1]/M;
        }
    }

    gpu_fft_execute(fft_pass[0]);

    // Write output data
    for (r=0; r<N; r++) {
        row = GPU_FFT_ROW(fft_pass[0], out, r);
        for (c=0; c<M; c++) {
            output_array[(r*M+c)] = row[c].re;
            // output_array[2*(r*M+c)] = row[c].re;
            // output_array[2*(r*M+c)+1] = row[c].im;
        }
    }

    // Clean-up properly.  Videocore memory lost if not freed !
    gpu_fft_release(fft_pass[0]);

    return 0;
}


int fft2d(uint32_t N, uint32_t M, float *input_array, float *output_array) {
    
    int r, c, ret, mb = mbox_open();

    int log2_N = log2_32(N);
    int log2_M = log2_32(M);

    struct GPU_FFT_COMPLEX *row;
    struct GPU_FFT_TRANS *trans;
    struct GPU_FFT *fft_pass[2];


    // Prepare 1st FFT pass
    ret = gpu_fft_prepare(mb, log2_M, GPU_FFT_FWD, N, fft_pass+0);
    if (ret) {
        return ret;
    }

    // Prepare 2nd FFT pass
    ret = gpu_fft_prepare(mb, log2_N, GPU_FFT_FWD, M, fft_pass+1);
    if (ret) {
        gpu_fft_release(fft_pass[0]);
        return ret;
    }

    // Transpose from 1st pass output to 2nd pass input
    ret = gpu_fft_trans_prepare(mb, fft_pass[0], fft_pass[1], &trans);
    if (ret) {
        gpu_fft_release(fft_pass[0]);
        gpu_fft_release(fft_pass[1]);
        return ret;
    }

    // Setup input data
    for (r=0;r<N;r++){
        row = GPU_FFT_ROW(fft_pass[0], in, r);
        for (c=0;c<M;c++){
            row[c].re = input_array[(r*M+c)];
            row[c].im = 0; // for now it will only accept real numbers as input
        }
    }

    // It's a FFT2D using FFT1D
    // ==> FFT() ==> T() ==> FFT()
    gpu_fft_execute(fft_pass[0]);
    gpu_fft_trans_execute(trans);
    gpu_fft_execute(fft_pass[1]);


    // Write output data
    for (r=0; r<M; r++) {
        row = GPU_FFT_ROW(fft_pass[1], out, r);
        for (c=0; c<N; c++) {
            output_array[2*(c*M+r)] = row[c].re;
            output_array[2*(c*M+r)+1] = row[c].im;
        }
    }

    // Clean-up properly.  Videocore memory lost if not freed !
    gpu_fft_release(fft_pass[0]);
    gpu_fft_release(fft_pass[1]);
    gpu_fft_trans_release(trans);

    return 0;
}

int ifft2d(uint32_t N, uint32_t M, float *input_array, float *output_array) {
    
    int r, c, ret, mb = mbox_open();

    int log2_N = log2_32(N);
    int log2_M = log2_32(M);
    long int NORM = N*M;

    struct GPU_FFT_COMPLEX *row;
    struct GPU_FFT_TRANS *trans;
    struct GPU_FFT *fft_pass[2];


    // Prepare 1st FFT pass
    ret = gpu_fft_prepare(mb, log2_M, GPU_FFT_REV, N, fft_pass+0);
    if (ret) {
        return ret;
    }
    
    // Prepare 2nd FFT pass
    ret = gpu_fft_prepare(mb, log2_N, GPU_FFT_REV, M, fft_pass+1);
    if (ret) {
        gpu_fft_release(fft_pass[0]);
        return ret;
    }

    // Transpose from 1st pass output to 2nd pass input
    ret = gpu_fft_trans_prepare(mb, fft_pass[0], fft_pass[1], &trans);
    if (ret) {
        gpu_fft_release(fft_pass[0]);
        gpu_fft_release(fft_pass[1]);
        return ret;
    }

    // Setup input data
    for (r=0;r<N;r++){
        row = GPU_FFT_ROW(fft_pass[0], in, r);
        for (c=0;c<M;c++){
            row[c].re = input_array[2*(r*M+c)];
            row[c].im = input_array[2*(r*M+c)+1];
        }
    }


    // It's a FFT2D using FFT1D
    // ==> FFT() ==> T() ==> FFT() ==>
    gpu_fft_execute(fft_pass[0]);
    gpu_fft_trans_execute(trans);
    gpu_fft_execute(fft_pass[1]);

    // Write output data
    for (r=0; r<M; r++) {
        row = GPU_FFT_ROW(fft_pass[1], out, r);
        for (c=0; c<N; c++) {
            output_array[(c*M+r)] = row[c].re/NORM;
            // output_array[2*(c*M+r)] = row[c].re/NORM;
            // output_array[2*(c*M+r)+1] = -row[c].im/NORM;
        }
    }

    // Clean-up properly.  Videocore memory lost if not freed !
    gpu_fft_release(fft_pass[0]);
    gpu_fft_release(fft_pass[1]);
    gpu_fft_trans_release(trans);

    return 0;
}