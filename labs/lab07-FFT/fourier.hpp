#ifndef FOURIER_H
#define FOURIER_H

using namespace std;
typedef complex<double> cd;
const TYPE PI = acos(-1);

void naive_dft(TYPE* x, TYPE* X_real, TYPE* X_imag, int N);
void fft_recursive(TYPE* x, TYPE* X_real, TYPE* X_imag, int N);
void fft_iterative(TYPE* a, TYPE *A, TYPE *X_real, TYPE *X_imag, int n);

int read_array(TYPE** x)
{    
    std::fstream in("signal.dat");
    int N = std::count(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>(), '\n');
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


// original DFT
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


// FFT Recursive
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

// utility function for integer log base 2
int mylog2(int x){
    int r = 0;
    if(x == 0){
        return -1;
    }
    while(x>1){
        x = x>>1;
        r++;
    }
    return r;
}

// Utility function for reversing the bits
// of given index x
unsigned int bitReverse(unsigned int x, int log2n){
    int n = 0;
    for (int i = 0; i < log2n; i++){
        n <<= 1;
        n |= (x & 1);
        x >>= 1;
    }
    return n;
}

// Iterative FFT function to compute the DFT of given coefficient vector
void fft_iterative(TYPE* a, TYPE *X_real, TYPE *X_imag, int n)
{
    int log2n = mylog2(n);
    // bit reversal of the given array
    for (unsigned int i = 0; i < n; ++i) {
        int rev = bitReverse(i, log2n);
        X_real[i] = a[rev];
        X_imag[i] = 0;
    }
    // j is iota
    for (int s = 1; s <= log2n; ++s) {
        int m = 1 << s; // 2 power s
        int m2 = m >> 1; // m2 = m/2 -1
        TYPE wr = 1.0;
        TYPE wi = 0.0;
        // principle root of nth complex
        // root of unity.
        TYPE wmr = cos(PI/m2);
        TYPE wmi = sin(PI/m2);

        for(int j = 0; j < m2; ++j) {
            for(int k = j; k < n; k += m){
                // t = twiddle factor
                TYPE tr = wr*X_real[k+m2] - wi*X_imag[k+m2];
                TYPE ti = wr*X_imag[k+m2] + wi*X_real[k+m2];
                TYPE ur = X_real[k];
                TYPE ui = X_imag[k];

                // similar calculating y[k]
                X_real[k] = ur + tr;
                X_imag[k] = ui + ti;

                // similar calculating y[k+n/2]
                X_real[k + m2] = ur - tr;
                X_imag[k + m2] = ui - ti;
            }
            double aux1=wr;
            wr = wr*wmr - wi*wmi;
            wi = aux1*wmi + wi*wmr;
        }
    }
}
#endif
