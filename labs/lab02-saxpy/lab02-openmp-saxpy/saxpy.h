#pragma once
// lab02-openmp-saxpy. Puede usar guanaco siempre que este disponible.
// SAXPY es un ejemplo HPC sencillo ==> Hola Mundo de HPC
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
// (6) Saque sus conclusiones sobre el rendimiento obtenido en base a ambos graficos.
void init_vec(int *a, int n, int c){
	#pragma omp parallel for
	for(int i=0; i<n; ++i){
		a[i] = c*i;
	}
}

// (1) funcion saxpy1 con parallel for
void saxpy1(int *s, int *x, int *y, int n, int a, int cs){
}

// (2) funcion saxpy2 con parallel for y chunksize = 1
void saxpy2(int *s, int *x, int *y, int n, int a, int cs){
}

// (3) funcion saxpy3 manual con particion segmentos continuos
void saxpy3(int *s, int *x, int *y, int n, int a, int nt){
}

// (4) funcion saxpy4 manual con particion accesos intercalados 
void saxpy4(int *s, int *x, int *y, int n, int a, int nt){
}

void print_vec(int *a, int n, const char *msg){
    if(n > 32){ return; }
    printf("%s\n[", msg);
    for(int i=0; i<n; ++i){
        printf("%i ", a[i]);
    }
    printf("]\n");
}
