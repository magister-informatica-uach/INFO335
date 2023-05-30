#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <bits/stdc++.h>
#include <math.h>
#include <omp.h>

#define TYPE double
using namespace std;
const TYPE pi = acos(-1);

// custom includes
#include "main.h"

int main(int argc, char** argv){
    if(argc != 4){
        fprintf(stderr, "Ejecutar como ./prog <n> <nt> <mode>\n-nt: num CPU threads\n-mode=0: DFT basico\n-mode=1: FFT Radix-2-rec \n-mode=2 FFT Radix-2-it\n");
        return EXIT_FAILURE;
    }
    int N = atoi(argv[1]);
    int nt = atoi(argv[2]);
    int mode = atoi(argv[3]);
    double t1, t2;
    printf("FFT N=%i nt=%i mode=%i\n", mode, nt, N); fflush(stdout);
    printf("Gen Random Signal......"); fflush(stdout);
    t1 = omp_get_wtime();
    TYPE* signal = genRandomSignal(N);
    t2 = omp_get_wtime();
    printf("done: %f secs\n", t2-t1); fflush(stdout);
    TYPE* fft_real = (TYPE*)malloc(N*sizeof(TYPE));
    TYPE* fft_imag = (TYPE*)malloc(N*sizeof(TYPE));
    
    t1 = omp_get_wtime();
    switch (mode){
        case 0: printf("DFT Algorithm.........."); fflush(stdout); dft(signal, fft_real, fft_imag, N); break;
        case 1: printf("FFT Recursive.........."); fflush(stdout); fft_recursive(signal, fft_real, fft_imag, N); break;
        case 2: printf("FFT Iterative.........."); fflush(stdout); fft_iterative(signal, fft_real, fft_imag, N); break;
    }
    t2 = omp_get_wtime();
    printf("done: %f secs\n", t2-t1); fflush(stdout);
    free(fft_real);
    free(fft_imag);
    free(signal);
    return 0;
}



