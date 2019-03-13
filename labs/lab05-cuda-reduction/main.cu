#include <cstdio>
#include <ctime>
#include <cuda.h>
#include <omp.h>
#include "cpured.h"
#include "gpured.h"

int main(int argc, char **argv){
    if(argc != 3){
        fprintf(stderr, "run as ./prog n m\nn: problem size\nm: mode (0: gpu, 1+ openmp with m threads)\n");
        exit(EXIT_FAILURE);
    }
    srand(time(NULL));
    int n = atoi(argv[1]); 
    int m = atoi(argv[2]); 
    int *a = (int*)malloc(sizeof(int)*n);

    // gen numbers
    gen_random_array(a, n);
    print_array(a, n, "input a[]");
    int ompred = omp_reduction(a, n, m);
    int cudared = cuda_reduction(a, n);

    printf("RESULTS\nCPU reduction = %i\nGPU reduction = %i\n\n", ompred, cudared);
}
