#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#define BSIZE2D 32

// EJERCICIOS
// (1) Implemente el matmul de GPU basico
// (2) Implemente el matmul de GPU usando memoria compartida
// (3) Compare el rendimiento de GPU Matmul vs el de CPU que hizo previamente

// GPU matmul basico
__global__ void kernel_matmul(int n, float *a, float *b, float *c){
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	float sum = 0.0f;
	for(int k=0; k<n; ++k){
		sum += a[ty*n + k]*b[k*n + tx];
	}
	c[ty*n + tx] = sum;
}

// GPU matmul shared memory 
__global__ void kernel_matmulsm(int n, float *a, float *b, float *c){
	__shared__ float as[BSIZE2D*BSIZE2D];	
	__shared__ float bs[BSIZE2D*BSIZE2D];	
	__shared__ float cs[BSIZE2D*BSIZE2D];	

	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	int ltx = threadIdx.x;
	int lty = threadIdx.y;

        cs[lty*BSIZE2D + ltx] = 0;
	// (1) hacer 'k' veces la version bloque 
	for(int k=0; k<n; k=k+BSIZE2D){
	// 	(a) cargar datos en as,bs. Escribir resultados en cs
	//	(b) sincorinizar la carga en as, bs, antes de calcular cs.
		as[lty*BSIZE2D + ltx] = a[ty*n + (k + ltx)];
		bs[lty*BSIZE2D + ltx] = b[(k + lty)*n + tx];
		__syncthreads();
		for(int r=0; r<BSIZE2D; ++r){
			cs[lty*BSIZE2D + ltx] += as[lty*BSIZE2D + r]*bs[r*BSIZE2D + ltx];
		}
		__syncthreads();
	}
	// (2) escribir cs en c
        c[ty*n + tx] = cs[lty*BSIZE2D + ltx];
}

void matrandom(int n, float *m){
    srand(1);
    for(int i=0; i<n; ++i){
        for(int j=0; j<n; ++j){
            m[i*n + j] = (float)rand()/((float)RAND_MAX);
            //m[i*n + j] = i;
        }
    }
}

void printmat(int n, float *m, const char* msg){
    printf("%s\n", msg);
    for(int i=0; i<n; ++i){
        for(int j=0; j<n; ++j){
            printf("%.2f ", m[i*n + j]);
        }
        printf("\n");
    }
}

int verify(int n, float *a, float *b, float *c, float *cgold){
    float error = 0.01f;
    for(int i=0; i<n; ++i){
        for(int j=0; j<n; ++j){
            float sum = 0.0f;
            for(int k=0; k<n; ++k){
                sum += a[i*n + k]*b[k*n + j];
            }
            cgold[i*n + j] = sum;
            if(fabs(c[i*n + j] - cgold[i*n + j]) >= error){
                fprintf(stderr, "error: c[%i][%i] ---> c %f    cgold %f\n", i, j, c[i*n+j], cgold[i*n+j]);
                return 0;
            }
        }
    }
    return 1;
}

int main(int argc, char **argv){
    printf("GPU MATMUL\n");
    if(argc != 2){
        fprintf(stderr, "run as ./prog n\n");
        exit(EXIT_FAILURE);
    }
    int n = atoi(argv[1]);
    float msecs = 0.0f;

    // (1) creando matrices en host
    float *a = new float[n*n];
    float *b = new float[n*n];
    float *c = new float[n*n];
    float *cgold = new float[n*n];
    printf("initializing A and B......."); fflush(stdout);
    matrandom(n, a);
    matrandom(n, b);
    if(n < 64){
        printmat(n, a, "mat a");
        printmat(n, b, "mat b");
    }
    printf("ok\n"); fflush(stdout);

    // (2) dejando matrices en device
    float *ad, *bd, *cd;
    cudaMalloc(&ad, sizeof(float)*n*n);
    cudaMalloc(&bd, sizeof(float)*n*n);
    cudaMalloc(&cd, sizeof(float)*n*n);
    cudaMemcpy(ad, a, sizeof(float)*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(bd, b, sizeof(float)*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(cd, c, sizeof(float)*n*n, cudaMemcpyHostToDevice);

    // (3) ejecutar matmul en GPU
    printf("computing C = A x B........"); fflush(stdout);
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        dim3 block(BSIZE2D, BSIZE2D, 1);
        dim3 grid((n+BSIZE2D-1)/BSIZE2D, (n+BSIZE2D-1)/BSIZE2D, 1); 
        cudaEventRecord(start);
        kernel_matmul<<<grid, block>>>(n, ad, bd, cd);
        //kernel_matmulsm<<<grid, block>>>(n, ad, bd, cd);
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&msecs, start, stop);
    printf("ok: time: %f secs\n", msecs/1000.0f);

    // (4) copiar resultado a host
    printf("copying result to host....."); fflush(stdout);
    cudaMemcpy(c, cd, sizeof(float)*n*n, cudaMemcpyDeviceToHost);
    printf("ok\n"); fflush(stdout);

    if(n < 50){
        printmat(n, c, "mat c");
    }

    // (5) verificar resultado contra calculo en CPU
    printf("verifying result..........."); fflush(stdout);
    if(!verify(n, a, b, c, cgold)){
        fprintf(stderr, "error verifying result\n");
        exit(EXIT_FAILURE);
    }
    printf("ok\n");
    printf("done!\n");
    exit(EXIT_SUCCESS);
}
    
