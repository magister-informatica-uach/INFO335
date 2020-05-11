#include <cstdlib>
#include <cstdio>
#include <omp.h>
// lab02-openmp-saxpy. Puede usar guanaco siempre que este disponible.

// (1) haga un programa saxpy y mida el tiempo del calculo
// (2) introduzca paralelismo con OpenMP, de distintas formas
//      a) funcion saxpy1 con parallel for
//      b) funcion saxpy2 con parallel for y chucksize = 1
//      c) funcion saxpy3 manual con particion segmentos continuos
//      d) funcion saxpy4 manual con particion segmentos intercalados
// (3) experimente comparando el resultado de cada metodo a distintos n
// (4) Hacer un grafico tiempo vs n, usando todos los cores de la CPU
// (5) Hacer un grafico tiempo vs nt, fijando el problema en n = 10^8
// (6) saque conclusiones sobre el rendimiento obtenido en base a ambos graficos.
void init_vec(int *a, int n, int c){
	#pragma omp parallel for
	for(int i=0; i<n; ++i){
		a[i] = c;
	}
}

// (1) version omp parallel for (simple)
void saxpy1(int *s, int *x, int *y, int n, int a, int cs){
}
// (2) version omp parallel for (simple)
void saxpy2(int *s, int *x, int *y, int n, int a, int cs){
}
// (3) version omp parallel for (simple)
void saxpy3(int *s, int *x, int *y, int n, int a, int nt){
}
// (4) version omp parallel for (simple)
void saxpy4(int *s, int *x, int *y, int n, int a, int nt){
}



int main(int argc, char **argv){
    if(argc != 5){
        fprintf(stderr, "error, ejecutar como: ./prog N threads metodo chunksize\n");
        exit(EXIT_FAILURE);
    }
    unsigned long N = atoi(argv[1]);
    unsigned int nt = atoi(argv[2]);
    unsigned int m = atoi(argv[3]);
    unsigned int cs = atoi(argv[4]);
    omp_set_num_threads(nt);
    double t1=0.0, t2=0.0;
    // creacion de los vectores
    int *x = new int[N];
    int *y = new int[N];
    int *s = new int[N];
    // inicializar vectores
    init_vec(x, N, 1);
    init_vec(y, N, 2);
    init_vec(s, N, 0);
    t1 = omp_get_wtime();
    // calculo saxpy
    switch(m){
	    case 1:
		printf("saxpy1\n");
		saxpy1(s, x, y, N, 10, cs);
		break;
	    case 2:
		printf("saxpy2\n");
		saxpy2(s, x, y, N, 10, cs);
		break;
	    case 3:
		printf("saxpy3\n");
		saxpy3(s, x, y, N, 10, nt);
		break;
	    case 4:
		printf("saxpy4\n");
		saxpy4(s, x, y, N, 10, nt);
		break;
    }
    t2 = omp_get_wtime();
    printf("N=%i    threads=%i   %f secs\n", N, nt, t2-t1);
}
