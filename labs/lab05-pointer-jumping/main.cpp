#include <iostream>
#include <cstdio>
#include <omp.h>

void printarray(int *a, int n, const char *msg){
    if( n <= 32) {
        printf("%s:\n", msg);
        for(int i=0; i<n; ++i){
            printf("%i  ", i);
        }
        printf("\n");
        for(int i=0; i<n; ++i){
            printf("%i  ", a[i]);
        }
        printf("\n");
        printf("\n");
    }
}

// pointer jumping (algoritmo)
void pointer_jumping(int *p, int n){
}

void pointer_jumping_par(int *p, int n, int nt){
    // IMPLEMENTAR
}


int main(int argc, char **argv){
	if(argc != 4){
		fprintf(stderr, "./prog n nt mode\nmode: 0 = secuencial,   1 = paralelo\n");
		exit(EXIT_FAILURE);
	}
	int n = atoi(argv[1]);
	int nt = atoi(argv[2]);
	int mode = atoi(argv[3]);
	int *p = new int[n];
	omp_set_num_threads(nt);
    printf("inicializando %i valores.....", n); fflush(stdout);
    #pragma omp parallel for
	for(int i=0; i<n-1; ++i){
		p[i] = i+1;
	}
	p[n-1] = n-1;
    printf("done\n"); fflush(stdout);
	printarray(p, n, "arreglo original");
    printf("pointer-jumping....."); fflush(stdout);
    double t1 = omp_get_wtime();
    if(mode){
	    pointer_jumping_par(p, n, nt);
    }
    else{
	    pointer_jumping(p, n);
    }
    double t2 = omp_get_wtime();
    printf("done\n"); fflush(stdout);
	printarray(p, n, "solucion");
    printf("OK: %f secs\n", t2-t1);
	return 0;
}
