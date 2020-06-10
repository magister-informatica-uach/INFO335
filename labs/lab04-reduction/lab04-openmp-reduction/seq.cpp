#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
int main(int argc, char** argv){
    // genera n numeros aleatorios y los suma
    int N = atoi(argv[1]);
    double sum = 0.0, *x;
    // memoria para n 'doubles'
    x = (double*)malloc(sizeof(double)*N);

    // inicializar con n numeros aleatorios normalizados entre 0 y 1
    for(int i = 0; i < N; ++i){
        x[i] = (double)rand()/(double)RAND_MAX;
    }

    // calcular la suma (reduccion)
    for(int i = 0; i < N; ++i){
        sum += x[i];
    }
    free(x);
    printf("sum = %f\nDONE\n", sum);
}
