#include <cstdio>
#include <cstdlib>
#include <omp.h>

// 2) Funcion que calcula el prefix_sum de A[], el cual es un arreglo de 'n' elementos, donde se cumple
void prefix_sum(int *A, int *P, int n){
    // solucion mas basica
    for(int i=0; i<n; ++i){
        for(int k=0; k<=i; ++k){
            P[i] = P[i] + A[k];
        }
    }
    // solucion mas eficiente 
    /*
    if(n==0){ return; }
    int accum = A[0];
    P[0] = accum;
    for(int i=1; i<n; ++i){
        accum = accum + A[i];
        P[i] = accum;
    }
    */
}


void gen_random_A(int *A, int n, int seed){
    srand(seed);
    for(int i=0; i<n; ++i){
        A[i] = rand() % 9;
    }
}

void print_array(int *A, int n, const char *msg){
    if(n > 64){
        return;
    }
    printf("%s\n", msg);
    for(int i=0; i<n; ++i){
        printf("%i ", A[i]);
    }
    printf("\n");
}

int main(int argc, char** argv){
    //              (0)  (1) (2)
    // se ejecuta ./prog <n> <a>
    if(argc != 3){
        fprintf(stderr, "ejecutar como ./prog <n> <a>\n");
        exit(EXIT_FAILURE);
    }
    int n = atoi(argv[1]);
    int a = atoi(argv[2]);
    printf("n = %i,  a = %i\n", n, a);
    // 1) generar un arreglo A[] de 'n' enteros aleatorios
    int *A = (int*)malloc(n*sizeof(int));
    gen_random_A(A, n, 13);
    print_array(A, n, "Arreglo A");
    // 3) utilizar la funcion prefix_sum
    int *P = (int*)malloc(n*sizeof(int));
    printf("Calculando prefix_sum......"); fflush(stdout);
    double t1 = omp_get_wtime();
    prefix_sum(A, P, n);
    double t2 = omp_get_wtime();
    printf("done: %f secs\n", t2-t1); fflush(stdout);
    print_array(P, n, "Arreglo P");
    exit(EXIT_SUCCESS);
}
