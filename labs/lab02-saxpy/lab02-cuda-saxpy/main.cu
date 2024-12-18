#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <omp.h>

// (1) haga un programa saxpy en CUDA:
//      a) funcion saxpy con mapeo de threads intercalado usando n threads (1 thread por dato).
// (2) Grafique tiempo vs n.
// (3) Reporte conclusiones sobre el rendimiento obtenido.
// (4) Compare el rendimiento vs Saxpy OpenMP, cual es mas rapida? por cuanto?
// (5) Haga un grafico con un 'n' grande, de tiempo vs blocksize. Cual fue el mejor blocksize?

// Kernel
__global__ void mikernel(float a, float *x, float *y, float *s, int n){
    // TRABAJAR AQUI
}

void cpu(float a, float *x, float *y, float *z, int n);
void init_vec(float *a, int n, float c);
void print_vec(float *a, int n, const char *msg);

int main(int argc,char **argv){
	if(argc != 4){
		fprintf(stderr, "error ejecutar como ./prog n mode BSIZE\n");
		exit(EXIT_FAILURE);
	}
    int n, m, bs;
    // punteros version HOST
    float a = 1.0f, *x,  *y,  *s;
    // punteros de mem version DEVICE
    float   *dx, *dy, *ds;
    // obtener argumentos
    n = atoi(argv[1]);
    m = atoi(argv[2]);
    bs = atoi(argv[3]);

    // inicializar arreglos en Host (CPU)
    x = new float[n];
    y = new float[n];
    s = new float[n];
    init_vec(x, n, 1);
    print_vec(x, n, "vector x");
    init_vec(y, n, 2);
    print_vec(y, n, "vector y");
    init_vec(s, n, 0);

    // allocar memoria en device  (GPU)
    // cudaMalloc( puntero del puntero, bytes)
    cudaMalloc(&dx, sizeof(float) * n);
    cudaMalloc(&dy, sizeof(float) * n);
    cudaMalloc(&ds, sizeof(float) * n);

    // copiar de Host -> Device
    //cudaMemcpy(destino, origen, bytes, direccion)
    cudaMemcpy(dx, x, sizeof(float)*n, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y, sizeof(float)*n, cudaMemcpyHostToDevice);
    //cudaMemcpy(ds, s, sizeof(float)*n, cudaMemcpyHostToDevice)

    // block -> maximo un total de 1024 threads en un bloque.
    dim3 block(bs, 1, 1);
    // grid -->  block ->   threads
    // grid 1D (porque el vector es lineal)
    // el grid esta definido en numero de bloques para x, y, z
    dim3 grid((n + bs -1)/bs, 1, 1);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	printf("calculando...."); fflush(stdout);
	cudaEventRecord(start);
	if(m){
		printf("GPU\n"); fflush(stdout);
		mikernel<<<grid, block>>>(a, dx, dy, ds, n);
	}
	else{
		printf("CPU\n"); fflush(stdout);
		cpu(a, x, y, s, n);	
	}
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	if(m){
		cudaMemcpy(s, ds, sizeof(float)*n, cudaMemcpyDeviceToHost);
	}
	// calculo en GPU
	printf("ok: %f secs\n", milliseconds/1000.0f); fflush(stdout);
	print_vec(s, n, "vector S");		
}



void cpu(float a, float *x, float *y, float *s, int n){
    #pragma omp parallel for num_threads(12)
	for(int i=0;i<n;i++){
		s[i]=a*x[i]+y[i];
	}
}
void init_vec(float *a, int n, float c){
	#pragma omp parallel for num_threads(12)
	for(int i=0; i<n; ++i){
		a[i] = c*i;
	}
}

void print_vec(float *a, int n, const char *msg){
    if(n > 32){ return; }
    printf("%s\n[", msg);
    for(int i=0; i<n; ++i){
        printf("%f ", a[i]);
    }
    printf("]\n");
}

