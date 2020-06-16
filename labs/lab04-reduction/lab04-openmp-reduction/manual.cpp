#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <omp.h>

double parallel_reduction(double *x, long n, int nt){
    // programar en paralelo reduccion (OpenMP)
    double *results = new double[nt];
    #pragma omp parallel shared(results)
    {
        // FASE 1 ---> de n valores a nt resultados.
        int tid = omp_get_thread_num();
        results[tid] = 0;
        int segment = (n + nt -1)/nt;
        int start = segment*tid;
        int end = start + segment;
        // sumando segmentos en paralelo
        for(int i=start; i<n && i<end; ++i){
            results[tid] += x[i];
        }
        #pragma omp barrier
        // terminamos con sumas parciales en "results", nos olvidamos de x
        // FASE 2 ---> el proceso O(log n) --> terminamos 1 valor
        // de nt --> a 1 resultado gradualmente
        int workers = nt/2;
        while(workers > 0){
            if(tid < workers){
                results[tid] += results[tid + workers];
            }
            workers = workers/2;
            #pragma omp barrier
        }
        // resultado queda en results[0]
    }
    return results[0];
}

int main(int argc, char** argv){
    if(argc != 3){
	    fprintf(stderr, "run as ./prog n nt\n");
	    exit(EXIT_FAILURE);
    }
    int N = atoi(argv[1]);
    int nt = atoi(argv[2]);
    omp_set_num_threads(nt);
    double sum = 0.0, *x;

    // malloc e inicializacion
    x = (double*)malloc(sizeof(double)*N);
    for(int i = 0; i < N; ++i){
        x[i] = (double)rand()/(double)RAND_MAX;
    }

    // calculo secuencial
    double t1 = omp_get_wtime();
    for(int i = 0; i < N; ++i){
        sum += x[i];
    }
    double t2 = omp_get_wtime();
    double tseq = t2-t1;

    // calculo paralelo
    t1 = omp_get_wtime();
    double psum = parallel_reduction(x, N, nt);
    t2 = omp_get_wtime();
    double tpar = t2-t1;


    free(x);
    printf("sum = %f (%f secs)\npsum = %f (%f secs)\nDONE\n", sum, tseq, psum, tpar);
}
