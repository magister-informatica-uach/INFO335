#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <omp.h>
int main(int argc, char** argv){
    if(argc != 3){
        fprintf(stderr, "ejecutar como ./prog <N> <nt>\n\n");
        exit(EXIT_FAILURE);
    }
    // genera n numeros aleatorios y los suma
    int N = atoi(argv[1]);
    int nt = atoi(argv[2]);
    omp_set_num_threads(nt);
    printf("N=%i  nt=%i\n", N, nt);
    double sum = 0.0, *x;
    double r[nt];
    // memoria para n 'doubles'
    x = (double*)malloc(sizeof(double)*N);

    // inicializar con n numeros aleatorios normalizados entre 0 y 1
    srand(13);
    printf("Creando %i numeros aleatorios......", N); fflush(stdout);
    double t1 = omp_get_wtime();
    for(int i = 0; i < N; ++i){
        x[i] = (double)rand()/(double)RAND_MAX;
    }
    double t2 = omp_get_wtime();
    printf("LISTO: %f secs\n", t2-t1); fflush(stdout);

    // calcular la suma (reduccion) en paralelo
    printf("Calculando Reduccion......"); fflush(stdout);
    t1 = omp_get_wtime();
    #pragma omp parallel for reduction(+:sum)
    for(int i = 0; i < N; ++i){
        sum += x[i];
    }
    //#pragma omp parallel
    //{
    //   // FASE 1, reduccion bloques secuenciales
    //   int tid = omp_get_thread_num();
    //   int chunk = (N + nt -1)/nt;
    //   int start = tid*chunk;
    //   int end = start + chunk; // (tid+1)*chunk
    //   double rlocal = 0.0;
    //   for(int i=start; i<N && i<end; ++i){
    //        rlocal += x[i];
    //   }
    //   // tenemos un bloque reducido (rlocal)
    //   // ubicarlo en el arreglo de resultados
    //   r[tid] = rlocal;
    //   #pragma omp barrier
    //   // FASE 2 (ASUMAMOS QUE nt es potencia de 2)
    //   int workers = nt/2;
    //   while(workers > 0){
    //       if(tid < workers){
    //            r[tid] += r[tid+workers];
    //       } 
    //       #pragma omp barrier
    //       workers = workers >> 1;
    //   }
    //}
    //sum = r[0];
    t2 = omp_get_wtime();
    printf("LISTO: %f secs\n", t2-t1); fflush(stdout);
    free(x);
    printf("sum = %f\nDONE\n", sum);
}
