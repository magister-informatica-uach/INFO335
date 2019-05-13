#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <math.h>

#define TYPE double
using namespace std;
const TYPE pi = acos(-1);

int read_array(TYPE** x);
void naive_dft(TYPE* x, TYPE* X_real, TYPE* X_imag, int N);
void fft_iterative(TYPE* x, TYPE* X_real, TYPE* X_imag, int N);
void fft_recursive(TYPE* x, TYPE* X_real, TYPE* X_imag, int N);

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        fprintf(stderr, "Ejecutar como ./prog mode\n-mode=0: naive DFT\n-mode=1 Radix-2 FFT\n-mode=2 Cooley-Tuckey FFT\n");
        exit(EXIT_FAILURE);
    }
    int mode = atoi(argv[1]);
    TYPE* signal;
    int N = read_array(&signal);
    TYPE* fft_real = (TYPE*)malloc(N*sizeof(TYPE));
    TYPE* fft_imag = (TYPE*)malloc(N*sizeof(TYPE));
    
    naive_dft(signal, fft_real, fft_imag, N);
    switch (mode)
    {
        case 0: naive_dft(signal, fft_real, fft_imag, N); break;
        case 1: fft_recursive(signal, fft_real, fft_imag, N); break;
        case 2: fft_iterative(signal, fft_real, fft_imag, N); break;
    }
    
    
    std::ofstream out("spectrum.dat");
    out.precision(10);
    for(int k=0;k<N;k++)
        out << fft_real[k] << "\t" << fft_imag[k] << "\n";
    
    free(fft_real);
    free(fft_imag);
    free(signal);
    return 0;
}



void fft_iterative(TYPE* x, TYPE* X_real, TYPE* X_imag, int N)
{
    //http://www.cplusplus.com/forum/general/171004/
    //https://dsp.stackexchange.com/questions/8804/bit-reverse-order-technique-in-fft

}

void fft_recursive(TYPE* x, TYPE* X_real, TYPE* X_imag, int N)
{
	if (N==16)
	{
		//N = 1
		//X_real[0] = x[0];
		//X_imag[0] = 0.0;
		naive_dft(x, X_real, X_imag, N);
	}
	else
	{
	TYPE *x_even = (TYPE*)malloc(sizeof(TYPE)*N/2);
	TYPE *x_odd = (TYPE*)malloc(sizeof(TYPE)*N/2);
	TYPE *X_real_even = (TYPE*)malloc(sizeof(TYPE)*N/2);
	TYPE *X_imag_even = (TYPE*)malloc(sizeof(TYPE)*N/2);
	TYPE *X_real_odd = (TYPE*)malloc(sizeof(TYPE)*N/2);
	TYPE *X_imag_odd = (TYPE*)malloc(sizeof(TYPE)*N/2);
	
	for(int n=0; n<N/2; n++)
	{
		x_even[n] = x[2*n];
		x_odd[n] = x[2*n+1];
	}
	fft_recursive(x_even, X_real_even, X_imag_even, N/2);
	fft_recursive(x_odd, X_real_odd, X_imag_odd, N/2);
	TYPE w_real, w_imag, tmp;
	for(int k=0; k<N/2; k++)
	{
		w_real = cos(2.0*pi*k/(TYPE)N);
	        w_imag = -sin(2.0*pi*k/(TYPE)N);
		tmp = w_real*X_real_odd[k] - w_imag*X_imag_odd[k];
	        X_real[k] = X_real_even[k] + tmp;
	        X_real[k+N/2] = X_real_even[k] - tmp;
		tmp = w_real*X_imag_odd[k] +  w_imag*X_real_odd[k];	        
		X_imag[k] = X_imag_even[k] + tmp;
	        X_imag[k+N/2] = X_imag_even[k] - tmp;
	}
	free(x_even);
	free(x_odd);    
	free(X_real_even);
	free(X_real_odd);
	free(X_imag_even);
	free(X_imag_odd);
	}
}

void naive_dft(TYPE* x, TYPE* X_real, TYPE* X_imag, int N)
{
    for(int k=0; k<N; k++)
    {
        X_real[k] = 0.0;
        X_imag[k] = 0.0;
        for(int n=0;n<N;n++)
        {
            X_real[k] += x[n]*cos(2.0*pi*n*k/(TYPE)N);
            X_imag[k] -= x[n]*sin(2.0*pi*n*k/(TYPE)N);
        }
    }    
}

int read_array(TYPE** x)
{    
    std::fstream in("signal.dat");
    int N = std::count(std::istreambuf_iterator<char>(in),
            std::istreambuf_iterator<char>(), '\n');
    in.clear();
    in.seekg(0, ios::beg);
    std::string line;
    int n = 0;
    *x = (TYPE*)malloc(N*sizeof(TYPE));
    std::string str;
    while(std::getline(in, str))
    {
        #if TYPE == double
        (*x)[n] = std::stod(str);
        #else
        (*x)[n] = std::stof(str);
        #endif
        //printf("%f\t", x[n]);
        ++n;
        
    }
    in.close();    
    return N;

}
