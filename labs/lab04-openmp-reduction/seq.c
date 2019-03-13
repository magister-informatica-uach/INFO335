#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
int main(int argc, char** argv){
    int N = atoi(argv[1]);
    double sum = 0.0, *x;
    x = (double*)malloc(sizeof(double)*N);
    for(int i = 0; i < N; ++i){
        x[i] = (double)rand()/(double)RAND_MAX;
    }
    for(int i = 0; i < N; ++i){
        sum += x[i];
    }
    free(x);
    printf("sum = %f\nDONE\n", sum);
}
// gcc main.c -o omp03redseq
