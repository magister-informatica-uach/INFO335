#include <cstdlib>
#include <cstdio>
#include <omp.h>

// lab02-openmp-saxpy. Puede usar guanaco siempre que este disponible.
// MORALEJA: en CPU multicore, conviene asignar segmentos continuos de memoria a los threads.
//  => localidad    < t0 >  < t1 >  .... < tn > 
//
// saxpy es un ejemplo HPC sencillo ==> Hola Mundo de HPC
// SAXPY => S = aX + Y   S, X, Y vectores de n dimensiones
// SAXPY se usa porque es sencillo, como hola mundo, pero no explota la CPU al 100%.
// SAXPY es de los ejemplos mas sencillos que ofrece gran paralelismo de datos.

// (1) haga un programa saxpy y mida el tiempo del calculo
// (2) introduzca paralelismo con OpenMP, de distintas formas
//      a) funcion saxpy1 con parallel for
//      b) funcion saxpy2 con parallel for y chunksize = 1
//      c) funcion saxpy3 con omp parallel manual con particion segmentos continuos
//      d) funcion saxpy4 con omp parallel manual con particion segmentos intercalados
// (3) experimente comparando el resultado de cada metodo a distintos n
// (4) Hacer un grafico tiempo vs n, usando todos los cores de la CPU
// (5) Hacer un grafico tiempo vs nt, fijando el problema en n = 10^8
// (6) saque conclusiones sobre el rendimiento obtenido en base a ambos graficos.
void init_vec(int *a, int n, int c){
	#pragma omp parallel for
	for(int i=0; i<n; ++i){
		a[i] = c*i;
	}
}

// (1) funcion saxpy1 con parallel for
void saxpy1(int *s, int *x, int *y, int n, int a, int cs){
    // directiva que paraleliza el for que viene a continuacion
    // al definir chunksize, este se calcula como chunksize = n / nt
    #pragma omp parallel for
    for(int i=0; i<n; ++i){
        s[i] = a*x[i] + y[i];
    }
}

// (2) funcion saxpy2 con parallel for y chucksize = 1
void saxpy2(int *s, int *x, int *y, int n, int a, int cs){
    //                                                          [ thread 1 | thread2 | thread1 ]
    // chunksize 1 con dos threads--> [...................] ==> [    x1    |    x2   |   x3    ]
    //                                                          [  chunk1  | chunk2  | chunk3  ]
    // fenomeno negativo al rendimiento => 'false sharing' el algoritmo gatilla la actualizacion de caches de
    // cada core cuando no es necesario.
    #pragma omp parallel for schedule(static,cs) 
    for(int i=0; i<n; ++i){
        s[i] = a*x[i] + y[i];
    }
}

// (3) funcion saxpy3 manual con particion segmentos continuos
// x1 x2 x3 x4 x5 x6 x7 x8 x9
// t1 t1 t1 t2 t2 t2 t3 t3 t3
void saxpy3(int *s, int *x, int *y, int n, int a, int nt){
    #pragma omp parallel shared(s, x, y, n, a, nt)
    {
        // i) Problema sin escenarios de "race conditions" o competencia por algun recurso.
        // ii) s -> escritura
        // iii) x, y, n, a, nt -> letura (nunca hay problemas)
        // iv) se activan todos los 'nt' threads
        // STRATEGIA:
        // a) calcular donde comienza a trabajar cada thread, y cuanto.
        // b) definir rangos unicos para cada thread (en el i-esimo thread).
        int tid = omp_get_thread_num();
        // techo(n/nt)
        int subsize = (n + nt -1)/nt;
        // c) donde comienza cada thread
        int start = subsize * tid;
        // d) procesar el lote que corresponde
        // printf("thread %i  start  %i   subsize%i\n", tid, start, subsize);
        for(int i=start; i< start + subsize && i<n; ++i){
            s[i] = a*x[i] + y[i];
        }
    }
}

// (4) funcion saxpy4 manual con particion accesos intercalados 
// x1 x2 x3 x4 x5 x6 x7 x8 x9
// t1 t2 t3 t1 t2 t3 t1 t2 t3
void saxpy4(int *s, int *x, int *y, int n, int a, int nt){
    #pragma omp parallel shared(s, x, y, n, a, nt)
    {
        // i) Problema sin escenarios de "race conditions" o competencia por algun recurso.
        // ii) s -> escritura
        // iii) x, y, n, a, nt -> letura (nunca hay problemas)
        // iv) se activan todos los 'nt' threads
        // STRATEGIA:
        // a) calcular donde comienza a trabajar cada thread, y cuanto.
        // b) definir rangos unicos para cada thread (en el i-esimo thread).
        int tid = omp_get_thread_num();
        // techo(n/nt)
        //int subsize = (n + nt -1)/nt;
        // c) donde comienza cada thread
        //int start = subsize * tid;
        // d) procesar el lote que corresponde
        // printf("thread %i  start  %i   subsize%i\n", tid, start, subsize);
        // x1 x2 x3 x4 x5 x6 x7 x8 x9 x10
        // t0 t1 t2 t0 t1 t2 t0 t1 t2  t0  t1
        for(int i=tid; i < n; i = i + nt){
            s[i] = a*x[i] + y[i];
        }
    }
}

void print_vec(int *a, int n, const char *msg){
    if(n > 32){ return; }
    printf("%s\n[", msg);
    for(int i=0; i<n; ++i){
        printf("%i ", a[i]);
    }
    printf("]\n");
}

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
