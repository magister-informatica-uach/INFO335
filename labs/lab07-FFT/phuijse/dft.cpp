#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

#define TYPE double
using namespace std;
const TYPE pi = std::acos(-1);

std::vector<TYPE> read_array();
void naive_dft(std::vector<TYPE> x, TYPE* X_real, TYPE* X_imag, int N);
void fft(std::vector<TYPE> x, TYPE* X_real, TYPE* X_imag, int N);

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        fprintf(stderr, "Ejecutar como ./prog mode\n-mode=0: naive DFT\n-mode=1 Cooley-Tuckey FFT\n");
        exit(EXIT_FAILURE);
    }
    int mode = atoi(argv[1]);
    std::vector<TYPE> signal = read_array();
    int N = signal.size();
    TYPE* fft_real = (TYPE*)malloc(N*sizeof(TYPE));
    TYPE* fft_imag = (TYPE*)malloc(N*sizeof(TYPE));
    switch (mode)
    {
        case 0: naive_dft(signal, fft_real, fft_imag, N); break;
        case 1: fft(signal, fft_real, fft_imag, N); break;
    }
    
    
    std::ofstream out("spectrum.dat");
    out.precision(10);
    for(int k=0;k<N;k++)
        out << fft_real[k] << "\t" << fft_imag[k] << "\n";
    
    free(fft_real);
    free(fft_imag);
    return 0;
}

void fft(std::vector<TYPE> x, TYPE* X_real, TYPE* X_imag, int N)
{
    //http://www.cplusplus.com/forum/general/171004/
    //https://dsp.stackexchange.com/questions/8804/bit-reverse-order-technique-in-fft
    TYPE Xe_real, Xe_imag;
    TYPE Xo_real, Xo_imag;
    TYPE w_real, w_imag;
    for(int k=0; k<N/2; k++)
    {
        Xe_real = 0.0;
        Xo_real = 0.0;
        Xe_imag = 0.0;
        Xo_imag = 0.0;
        for(int n=0;n<N/2;n++)
        {
            w_real = std::cos(2.0*pi*2*n*k/(TYPE)N);
            w_imag = std::sin(2.0*pi*2*n*k/(TYPE)N);
            Xe_real += x[2*n]*w_real;
            Xe_imag -= x[2*n]*w_imag;
            Xo_real += x[2*n+1]*w_real;
            Xo_imag -= x[2*n+1]*w_imag;
        }
        w_real = std::cos(2.0*pi*k/(TYPE)N);
        w_imag = -std::sin(2.0*pi*k/(TYPE)N);
        X_real[k] = Xe_real + w_real*Xo_real - w_imag*Xo_imag;
        X_imag[k] = Xe_imag + w_real*Xo_imag + w_imag*Xo_real;
        X_real[k+N/2] = Xe_real - w_real*Xo_real + w_imag*Xo_imag;
        X_imag[k+N/2] = Xe_imag - w_real*Xo_imag - w_imag*Xo_real;

    }

}
void naive_dft(std::vector<TYPE> x, TYPE* X_real, TYPE* X_imag, int N)
{
    for(int k=0; k<N; k++)
    {
        X_real[k] = 0.0;
        X_imag[k] = 0.0;
        for(int n=0;n<N;n++)
        {
            X_real[k] += x[n]*std::cos(2.0*pi*n*k/(TYPE)N);
            X_imag[k] -= x[n]*std::sin(2.0*pi*n*k/(TYPE)N);
        }
    }    
}
std::vector<TYPE> read_array()
{    
    std::fstream in("signal.dat");
    int N = std::count(std::istreambuf_iterator<char>(in),
            std::istreambuf_iterator<char>(), '\n');
    in.clear();
    in.seekg(0, ios::beg);
    std::string line;
    int i = 0;
    std::vector<TYPE> signal(N);
    std::string str;
    while(std::getline(in, str))
    {
        signal[i] = std::stod(str);;
        ++i;
    }
    in.close();    
    return signal;

}


