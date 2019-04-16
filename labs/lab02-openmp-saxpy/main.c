#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

// (1) haga un programa saxpy y mida el tiempo del calculo
// (2) introduzca paralelismo con OpenMP, de distintas formas
//      a) funcion saxpy1 con parallel for
//      b) funcion saxpy2 con parallel for y chucksize = 1
//      c) funcion saxpy3 manual con particion segmentos continuos
//      d) funcion saxpy4 manual con particion segmentos intercalados
// (3) experimente comparando el resultado de cada metodo a distintos tamanos
// (4) saque conclusiones sobre el rendimiento obtenido, como escala con n y p
int main(int argc, char **argv){
    if(argc != 3){
        fprintf(stderr, "error, ejecutar como: ./prog N threads\n");
        exit(EXIT_FAILURE);
    }
    unsigned long N = atoi(argv[1]);
    int nt = 1;
    double t1=0.0, t2=0.0;
    printf("N=%i    threads=%i   %f secs\n", N, nt, t2-t1);
}
