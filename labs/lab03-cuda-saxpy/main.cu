#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <omp.h>

// (1) haga un programa saxpy en CUDA, de distintas formas
//      a) funcion saxpy1 manual con particion segmentos continuos
//      b) funcion saxpy2 manual con particion segmentos intercalados
// (2) experimente comparando el resultado de cada metodo a distintos tamanos
// (3) saque conclusiones sobre el rendimiento obtenido en funcion de n, p y el mapeo de hilos
// (4) compare el rendimiento de esta solucion vs el de CPU + OpenMP.

void imprime(float *a, int n);
void cpu(float a, float *x, float *y, float *z, int n);
__global__ void mikernel(float a, float *x, float *y, float *z, int n){

}

int main(int argc,char **argv){
	if(argc != 4){
		fprintf(stderr, "error ejecutar como ./prog n mode BSIZE\n");
		exit(EXIT_FAILURE);
	}
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
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

