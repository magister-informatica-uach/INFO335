#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <omp.h>

void matmul(int *a, int *b, int *c, int n);
void block_matmul(int *a, int *b, int *c, int n, int bsize);
void initmat(int *m, int n, int val);
void initmati(int *m, int n);
int sqdiffmat(int *a, int *b, int n);
void printmat(int *a, int n, const char *name);
int diffmat(int *a, int *b, int n, const char *mesg);

int main(int argc, char **argv){
    if(argc != 4){
        fprintf(stderr, "run as ./prog n b printflag\n");
        exit(EXIT_SUCCESS);
    }
    double t1, t2, taux;
    int n = atoi(argv[1]);
    int bsize = atoi(argv[2]);
    int *a, *b, *c1, *c2;
    a = (int*)malloc(sizeof(int) * n * n);
    b = (int*)malloc(sizeof(int) * n * n);
    c1 = (int*)malloc(sizeof(int) * n * n);
    c2 = (int*)malloc(sizeof(int) * n * n);

    // init
    initmati(a, n);
    initmati(b, n);
    initmat(c1, n, 0);
    initmat(c2, n, 0);

    // basico
    taux = omp_get_wtime();
    matmul(a, b, c1, n);
    t1 = omp_get_wtime() - taux;

    // blocked lines
    taux = omp_get_wtime();
    block_matmul(a, b, c2, n, bsize);
    t2 = omp_get_wtime() - taux;


    if(atoi(argv[3]) == 1){
        printmat(c1, n, "c1 naive");
        printmat(c2, n, "c2 blocked");
        diffmat(c1, c2, n, "c1-c2");
    }

    printf("diff sum(c1[i] - c2[i])^2 = %i\n", sqdiffmat(c1, c2, n));
    printf("classic matmul t1 = %f\nblock matmul t2 = %f\n", t1, t2);
    return 0;
}

void matmul(int *a, int *b, int *c, int n){
    for(int i=0; i<n; ++i){
        for(int j=0; j<n; ++j){
            int val = c[i*n + j];
            for(int k=0; k<n; ++k){
                val += a[i*n+k] * b[k*n +j];
            }
            c[i*n + j] = val;
        }
    }
}

// implementar
void block_matmul(int *a, int *b, int *c, int n, int bsize){

}


void initmat(int *m, int n, int val){
    for(int i=0; i<n*n; ++i){
        m[i] = val;
    }
}

void initmati(int *m, int n){
    for(int i=0; i<n*n; ++i){
        m[i] = i;
    }
}

int sqdiffmat(int *a, int *b, int n){
    int diff = 0;
    for(int i=0; i<n; ++i){
        for(int j=0; j<n; ++j){
            diff += (a[i*n + j] - b[i*n + j])*(a[i*n + j] - b[i*n + j]);
        }
    }
    return diff;
}

int diffmat(int *a, int *b, int n, const char *mesg){
    int diff = 0;
    printf("%s:\n", mesg);
    for(int i=0; i<n; ++i){
        for(int j=0; j<n; ++j){
            printf(" %i   ", a[i*n + j] - b[i*n + j]);
        }
        printf("\n");
    }
    return diff;
}

void printmat(int *a, int n, const char *name){
    printf("mat %s:\n", name);
    for(int i=0; i<n; ++i){
        for(int j=0; j<n; ++j){
            printf("%i ", a[i*n + j]);
        }
        printf("\n");
    }
    printf("\n");
}


