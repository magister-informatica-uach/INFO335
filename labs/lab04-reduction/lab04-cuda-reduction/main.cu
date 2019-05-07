#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#define BSIZE 256

__global__ void kernel_initarray(float *a, long n){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n){
        a[tid] = tid;
    }
}

__global__ void kernel_reduction(float *a, long n){
    __shared__ float sdata[BSIZE];
    unsigned int ltid = threadIdx.x;
    unsigned int tid= blockIdx.x*blockDim.x+ threadIdx.x;
    sdata[ltid] = 0.0f;
    if(tid < n){ 
        sdata[ltid] = a[tid];
    }
    __syncthreads();
    int s = blockDim.x >> 1;
    while(s > 0){
        if(ltid < s){
            sdata[ltid] += sdata[ltid + s];
        }
        s = s >> 1;
        __syncthreads();
    }
    if(ltid == 0){
        atomicAdd(&a[0], sdata[0]);
    }
}


float cpu_reduction(float *a, long n){
    float sum = 0.0f;
    for(long i=0; i<n; ++i){
        sum += a[i];
    }
    return sum;
}

int main(int argc, char **argv){
    if(argc != 2){
        fprintf(stderr, "run as ./prog n\n");
        exit(EXIT_FAILURE);
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    long n = atoi(argv[1]);
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
    printf("GPU reduction........................"); fflush(stdout);
    cudaEventRecord(start);
    kernel_reduction<<<grid, block>>>(xd, n);
    cudaDeviceSynchronize();
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
    printf("CPU reduction........................"); fflush(stdout);
    cudaEventRecord(start);
    cpures = cpu_reduction(xh, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cputime, start, stop);
    printf("done\n"); fflush(stdout);
    printf("GPU result: %f  (%f secs)\nCPU result: %f  (%f secs)\n", gpures, gputime*0.001f, cpures, cputime*0.001f);




}
