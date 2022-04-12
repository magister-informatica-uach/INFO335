// MULTI-GPU
// - ES UN MODELO HYBRIDO DE MEMORIA
//      - MEMORIA COMPARTIDA A NIVEL DE NODO (RAM CPU)
//      - MEMORIA DISTRIBUIDA A NIVEL GPUS
//          - CADA GPU TIENE SU PROPIO ESPACIO DE MEMORIA FISICA
// - COMUNICACION SE PUEDE LOGRAR P2P GPU-GPU (SINO, A TRAVES DE CPU)
// - NO HAY PASO DE MENSAJES MPI, SINO QUE TRASPASOS DE MEMORIA POR PUNTEROS AL ESTILO MEMCPY

#include <cuda.h>
#include <omp.h>
#include <cstdio>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define SEQWORK 1000000

__global__ void kernelAddOne(int *m, int n){
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if(tid < n){
        for(int i = 0; i<SEQWORK; ++i){
            m[tid] = (m[tid] + 1) % 245000010;
        }
    }
}

// SWAP DE DATOS ENTRE GPU0 y GPU1
int main(int argc, char **argv){
    // 1) ARGUMENTOS
    if(argc != 3){
        fprintf(stderr, "run as ./prog n ngpus\n");
    }
    int n = atoi(argv[1]);
    int ngpus = atoi(argv[2]);
    omp_set_num_threads(ngpus);

    // 2) CREACION DE DATOS EN HOST
    printf("Creando arreglo de %i ints.......", n); fflush(stdout);
    int *arrayH = new int[n];
    for(int i=0; i<n; ++i){
        arrayH[i] = i;
    }
    printf("done\n"); fflush(stdout);


    // 3) ZONA OPENMP (cada thread se hace cargo de una GPU)
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nt = omp_get_num_threads();
        int chunk = n/nt;
        gpuErrchk(cudaSetDevice(tid));
        printf("[THREAD %i] Creando arreglo en GPU %i de %i ints\n", tid, tid, chunk);
        int *arrayGPU;
        cudaMalloc(&arrayGPU, sizeof(int)*chunk);
        cudaMemcpy(arrayGPU, arrayH + tid*chunk, sizeof(int)*chunk, cudaMemcpyHostToDevice);

        // 4) CALCULAR array[i] = array[i] + 1   en multi-GPU
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        dim3 block(1024, 1, 1);
        dim3 grid( (chunk + block.x - 1)/block.x, 1, 1);
        printf("[THREAD %i] kernel.......", tid); fflush(stdout);
        cudaEventRecord(start);
        kernelAddOne<<< grid, block >>>(arrayGPU, chunk);
        cudaDeviceSynchronize();
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float msecs;
        cudaEventElapsedTime(&msecs, start, stop);
        printf("[THREAD %i] done: time: %f secs\n", tid, msecs/1000.0f);
        cudaMemcpy(arrayH + tid*chunk, arrayGPU, sizeof(int)*chunk, cudaMemcpyDeviceToHost);
    }// barrera implicita
    
    // 5) IMPRIMIR RESULTADO
    if(n <= 64){
        for(int i=0; i<n; ++i){
            printf("A[i] = %i\n", arrayH[i]);
        }
    }
}
