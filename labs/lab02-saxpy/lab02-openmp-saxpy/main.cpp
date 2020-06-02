#include <cstdlib>
#include <cstdio>
#include <omp.h>

#include "saxpy.h"

int main(int argc, char **argv){
    if(argc != 5){
        fprintf(stderr, "error, ejecutar como: ./prog N nt metodo chunksize\n");
        exit(EXIT_FAILURE);
    }
    unsigned long N = atoi(argv[1]);
    unsigned int nt = atoi(argv[2]);
    unsigned int m = atoi(argv[3]);
    unsigned int cs = atoi(argv[4]);
    // funciones OpenMP comienzan con omp_...
    omp_set_num_threads(nt);
    double t1=0.0, t2=0.0;
    // creacion de los vectores
    int *x = new int[N];
    int *y = new int[N];
    int *s = new int[N];
    // inicializar vectores
    init_vec(x, N, 1);
    print_vec(x, N, "vector x");
    init_vec(y, N, 2);
    print_vec(y, N, "vector y");
    init_vec(s, N, 0);
    int alpha = 1;

    printf("calculando SAXPY con a=%i.......", alpha); fflush(stdout);
    t1 = omp_get_wtime();
    // calculo saxpy
    switch(m){
	    case 1:
		printf("saxpy1\n");
		saxpy1(s, x, y, N, alpha, cs);
		break;
	    case 2:
		printf("saxpy2\n");
		saxpy2(s, x, y, N, alpha, cs);
		break;
	    case 3:
		printf("saxpy3\n");
		saxpy3(s, x, y, N, alpha, nt);
		break;
	    case 4:
		printf("saxpy4\n");
		saxpy4(s, x, y, N, alpha, nt);
		break;
    }
    t2 = omp_get_wtime();
    printf("done\n"); fflush(stdout);
    print_vec(s, N, "vector S");
    printf("N=%i    threads=%i   %f secs\n", N, nt, t2-t1);
}
