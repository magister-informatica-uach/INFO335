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
#include "fourier.hpp"



int main(int argc, char** argv)
{
    if(argc != 2)
    {
        fprintf(stderr, "Ejecutar como ./prog mode\n-mode=0: naive DFT\n-mode=1 Radix-2 FFT\n-mode=2 Cooley-Tuckey FFT (iterative)\n");
        exit(EXIT_FAILURE);
    }
    int mode = atoi(argv[1]);
    TYPE* signal;
    int N = read_array(&signal);
    TYPE* fft_real = (TYPE*)malloc(N*sizeof(TYPE));
    TYPE* fft_imag = (TYPE*)malloc(N*sizeof(TYPE));
    
    printf("FFT mode=%i N=%i.......", mode, N); fflush(stdout);
    double t1 = omp_get_wtime();
    switch (mode)
    {
        case 0: naive_dft(signal, fft_real, fft_imag, N); break;
        case 1: fft_recursive(signal, fft_real, fft_imag, N); break;
        case 2: fft_iterative(signal, fft_real, fft_imag, N); break;
    }
    double t2 = omp_get_wtime();
    printf("done\n"); fflush(stdout);
    
    
    std::ofstream out("spectrum.dat");
    out.precision(10);
    for(int k=0;k<N;k++){
        out << fft_real[k] << "\t" << fft_imag[k] << "\n";
    }
    
    free(fft_real);
    free(fft_imag);
    free(signal);
    //free(signalrev);
    printf("time: %f secs\n", t2-t1);
    return 0;
}



