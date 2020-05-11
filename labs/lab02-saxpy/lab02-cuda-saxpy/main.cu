#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <omp.h>

// (1) haga un programa saxpy en CUDA, de distintas formas
//      a) funcion saxpy1 con mapeo de threads intercalado usando n/100 threads.
//      b) funcion saxpy2 con mapeo de threads continuo usando n/100 threads.
//      c) funcion saxpy3 con mapeo de threads intercalado usando n threads.
// (2) Experimente con cada metodo a distintos tamanos y descubra cual rinde mejor.
// (3) Grafique tiempo vs n, incluyendo curvas para a) b) c).
// (4) Reporte conclusiones sobre el rendimiento obtenido y contraste cada approach.
// (5) Compare el rendimiento vs Saxpy OpenMP, cual es mas rapida? por cuanto?

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

