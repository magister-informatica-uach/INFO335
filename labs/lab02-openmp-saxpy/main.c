#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

// (1) haga un programa saxpy y mida el tiempo del calculo
// (2) introduzca paralelismo con OpenMP 
// (3) grafique la aceleracion
int main(int argc, char **argv){
    if(argc != 3){
        fprintf(stderr, "error, ejecutar como: ./prog N threads\n");
        exit(EXIT_FAILURE);
    }
    unsigned long N = atoi(argv[1]);
    int nt = 1;
    double t1=0.0, t2=0.0;
    printf("N=%i    threads=%i   %f secs\n", N, nt, t2-t1);
}
