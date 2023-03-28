#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define BSIZE 256

// VALORES A SUMAR SON A[] = [0,1,2,3,4,5,...., n-1]
__global__ void kernel_initarray(float *a, long n){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n){
        a[tid] = tid;
    }
}

__global__ void kernel_reduction(float *a, long n){
    // AQUI PROGRAMAR SOLUCION GPU DE REDUCCION
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int half = (n+2-1)/2;
    if(tid < half){
        if(tid + half < n){
            a[tid] = a[tid] + a[tid + half];
        }
    }
}

float cpu_reduction(float *a, long n){
    float sum = 0.0f;
    // averiguen como transformar rapidamente esto en paralelo con OpenMP
    #pragma omp parallel for reduction(+:sum)
    for(long i=0; i<n; ++i){
        sum += a[i];
    }
    return sum;
}

void print_gpu_array(float *xd, long n, const char *msg){
    float *xh = new float[n];
    cudaMemcpy(xh, xd, sizeof(int)*n, cudaMemcpyDeviceToHost);
    printf("%s\n", msg);
    for(int i=0; i<n; ++i){
        printf("%f ", xh[i]);
    }
    printf("\n");
}

int main(int argc, char **argv){
    if(argc != 3){
        fprintf(stderr, "run as ./prog n nt\n");
        exit(EXIT_FAILURE);
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    long n = atoi(argv[1]);
    int nt = atoi(argv[2]);
    omp_set_num_threads(nt);
    float *xd, *xh;
    float gpures, cpures;
    float gputime, cputime;
    dim3 block(BSIZE, 1, 1);
    dim3 grid((n+BSIZE-1)/(BSIZE), 1, 1);
    xh = new float[n];
    cudaMalloc(&xd, sizeof(float)*n);

    // (1) parallel GPU init
    printf("GPU Init Array......................."); fflush(stdout);
    kernel_initarray<<<grid, block >>> (xd, n);
    cudaDeviceSynchronize();
    printf("done\n"); fflush(stdout);


    // (2) parallel reduction
    int naux=n, cont = 0;
    printf("GPU reduction........................"); fflush(stdout);
    cudaEventRecord(start);
    while(naux > 1){
        //printf("\n\nPASS %i, naux=%i\n", ++cont, naux);
        //print_gpu_array(xd, naux, "input");
        kernel_reduction<<<grid, block>>>(xd, naux);
        cudaDeviceSynchronize();
        naux = (naux+1)/2;
        //print_gpu_array(xd, naux, "result");
    }
    printf("done\n"); fflush(stdout);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gputime, start, stop);


    // (3) get result
    cudaMemcpy(&gpures, xd, sizeof(float), cudaMemcpyDeviceToHost);

    // (4) CPU result for validation
    printf("Init array and copy GPU -> CPU......."); fflush(stdout);
    kernel_initarray<<<grid, block >>> (xd, n);
    cudaMemcpy(xh, xd, sizeof(float)*n, cudaMemcpyDeviceToHost);
    printf("done\n"); fflush(stdout);
    printf("CPU reduction (%i threads)..........", nt); fflush(stdout);
    cudaEventRecord(start);
    cpures = cpu_reduction(xh, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cputime, start, stop);
    printf("done\n"); fflush(stdout);
    printf("GPU result: %f  (%f secs)\nCPU result: %f  (%f secs)\n", gpures, gputime*0.001f, cpures, cputime*0.001f);
}
