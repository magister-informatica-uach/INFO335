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

// analisis problema: subir la cota minima lo mas posible (Omega(f(n)))
// analisis algoritmo: bajar la cota superior de tiempo (O(f(n))
// pointer jumping (algoritmo)
void pointer_jumping(int *p, int n){
    // condicion de termino p[i] = p[p[i]] para todos
    int incompleto = 0;
    int count = 0;
    do{
        //printf("paso %i\n", count++);
        incompleto = 0;
        for(int i=0; i<n-1; ++i){
            if(p[i] != p[p[i]]){
                // jump
                p[i] = p[p[i]];
                incompleto = 1;
            }
        }
    }
    while(incompleto);
}

void pointer_jumping_par(int *p, int n, int nt){
    int incompleto = 0;
    #pragma omp parallel shared(incompleto)
    {
        // 1)  obtener inicio y fin de segmentos de trabajo
        int tid = omp_get_thread_num();
        int seg = (n+nt-1)/nt;
        int begin = tid*seg;
        int end = begin + seg;
        int c = 0;
        #pragma omp barrier
        do{
            //printf("paso %i\n", ++c);
            #pragma omp barrier
            if(tid == 0){
                incompleto = 0;
            }
            #pragma omp barrier
            for(int i=begin; i<end && i<n; ++i){
                if(p[i] != p[p[i]]){
                    p[i] = p[p[i]];
                    incompleto = 1;
                }
            }
            #pragma omp barrier
        }
        while(incompleto);
    }
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
