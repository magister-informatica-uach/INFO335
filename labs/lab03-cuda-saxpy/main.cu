#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <omp.h>

void imprime(float *a, int n);
void cpu(float a, float *x, float *y, float *z, int n);
__global__ void mikernel(float a, float *x, float *y, float *z, int n){
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	//printf("[GPU] tid %i n %i\n", tid, n);
	// thread coarsening
	if(tid < n){
		// fine grained
		z[tid] = a*x[tid] + y[tid];
	}
}

int main(int argc,char **argv){
	if(argc != 4){
		fprintf(stderr, "error ejecutar como ./prog n mode BSIZE\n");
		exit(EXIT_FAILURE);
	}
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int n=atoi(argv[1]);
	int m=atoi(argv[2]);
	int BSIZE=atoi(argv[3]);
	float *x,*y,*z;
	float *dx, *dy, *dz;
	float a = 1.0f;
	printf("inicializando...."); fflush(stdout);
	x=(float*)malloc(sizeof(float)*n);
	y=(float*)malloc(sizeof(float)*n);
	z=(float*)malloc(sizeof(float)*n);
	// FASE 1  copiar   CPU --->  GPU
	cudaMalloc(&dx, sizeof(float)*n);	
	cudaMalloc(&dy, sizeof(float)*n);	
	cudaMalloc(&dz, sizeof(float)*n);	
	// destino, origen
	for(int i=0;i<n;i++){ //inicializar vectores Z, X e Y
		x[i]=1.0f;
		y[i]=1.0f;
		z[i]=0.0f;
	}
	cudaMemcpy(dx, x, sizeof(float)*n, cudaMemcpyHostToDevice);
	cudaMemcpy(dy, y, sizeof(float)*n, cudaMemcpyHostToDevice);
	cudaMemcpy(dz, z, sizeof(float)*n, cudaMemcpyHostToDevice);
	printf("ok\n"); fflush(stdout);
	/*
	 x1 x2 x3 x4 ....... ..............xn
	[t1 t2 t3 ..tk] [t1 t2 t3 ... tk] 
	*/
	dim3 block(BSIZE, 1, 1);
	dim3 grid( (n + block.x - 1)/block.x, 1, 1);
	printf("block(%i, %i, %i)   grid(%i, %i, %i)\n", block.x, block.y, block.z, 
							 grid.x, grid.y, grid.z);
	printf("calculando...."); fflush(stdout);
	cudaEventRecord(start);
	if(m){
		printf("GPU\n"); fflush(stdout);
		mikernel<<<grid, block>>>(a, dx, dy, dz, n);
	}
	else{
		printf("CPU\n"); fflush(stdout);
		cpu(a, x, y, z, n);	
	}
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	if(m){
		cudaMemcpy(z, dz, sizeof(float)*n, cudaMemcpyDeviceToHost);
	}
	// calculo en GPU
	printf("ok: %f secs\n", milliseconds/1000.0f); fflush(stdout);
	if(n < 30){
		imprime(z, n);		
	}
}
void imprime(float *a, int n){
	for(int i=0; i<n; ++i){
		printf("z[%i] = %f\n", i, a[i]);
	}
}


void cpu(float a, float *x, float *y, float *z, int n){
	for(int i=0;i<n;i++){
		z[i]=a*x[i]+y[i];
	}
}

