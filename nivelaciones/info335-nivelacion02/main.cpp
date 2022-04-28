#include <cstdio>
#include <cstdlib>
#include <omp.h>

void fill_matrix_rand(int *M, int n, int seed){
    srand(seed);
    for(int i=0; i<n; ++i){
        for(int j=0; j<n; ++j){
            long index = i*n +j;
            M[index] = rand() % 9;
        }
    }
}

void fill_matrix_const(int *M, int n, const int c){
    for(int i=0; i<n; ++i){
        for(int j=0; j<n; ++j){
            long index = i*n +j;
            M[index] = c;
        }
    }
}

void print_matrix(int *M, int n, const char *msg){
    if(n > 32){
        return;
    }
    printf("%s\n", msg);
    for(int i=0; i<n; ++i){
        for(int j=0; j<n; ++j){
            long index = i*n +j;
            printf("%4i ", M[index]);
        }
        printf("\n");
    }
}

void matmul(int *A, int *B, int *C, int n){
    // explorar C
    for(int i=0; i<n; ++i){
        for(int j=0; j<n; ++j){
            // calcular C_ij
            int acc = 0;
            for(int k=0; k<n; ++k){
                acc += A[i*n+k]*B[k*n+j];
            }
            C[i*n+j] = acc;
        }
    }
}

int main(int argc, char** argv){
    // se ejecuta ./prog <n>
    // 1) tomar argumentos para saber 'n'
    if(argc != 2){
        fprintf(stderr, "ejecutar como ./prog <n>\n");
        exit(EXIT_FAILURE);
    }
    int n = atoi(argv[1]);
    size_t nelem = n*n;
    printf("Producto de matrices de %ix%i\n", n, n);

    // 2) construir matrices
    int *A = new int[nelem];
    int *B = new int[nelem];
    int *C = new int[nelem];
    fill_matrix_rand(A, n, 13);
    fill_matrix_rand(B, n, 15);
    fill_matrix_const(C, n, 0);

    print_matrix(A, n, "A");
    print_matrix(B, n, "B");
    print_matrix(C, n, "C");

    // 3) multiplicar matrices
    double t1 = omp_get_wtime();
    printf("Computing C = A x B........"); fflush(stdout);
    matmul(A, B, C, n);
    double t2 = omp_get_wtime();
    printf("done: %f secs\n", t2-t1);


    // 4) mostrar resultado (si n es pequeno)
    print_matrix(C, n, "C RESULT");
    exit(EXIT_SUCCESS);
}
