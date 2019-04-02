#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <algorithm>
#include <omp.h>
#include "tools.h"

using namespace std;

int main(int argc, char** argv){
    random_device r;
    // (1) obtener argumentos ( ./prog n grupo modo )
    if(argc != 4){
        fprintf(stderr, "\nrun as ./prog n grupo modo\ngrupo: num threads\nmodo = 0 sobrio    1 borracho\n\n");
        exit(EXIT_FAILURE);
    }
    int n = atoi(argv[1]);
    int grupo = atoi(argv[2]);
    int modo = atoi(argv[3]);


    // (2) definir y crear matriz con nino aleatorio
    long mtotal = (long)n*n;
    int *mat = new int[mtotal];
    printf("creando matriz con zeros....."); fflush(stdout);
    init_mat_const(mat, n, 0);
    printf("done\n"); fflush(stdout);
    long posnino =  r();
    mat[posnino % mtotal] = 1;


    // (3) mostrar matriz
    #ifdef DEBUG
        if(n < 50){
            print_mat(mat, n);
        }
    #endif


    // (4) buscar nino
    double t1 = omp_get_wtime();
    pair<int,int> npos;
    if(modo == 0){
        printf("buscando a lo sobrio........."); fflush(stdout);
        npos = search_seq_sobrio(mat, n, grupo);
        printf("done\n"); fflush(stdout);
    }
    if(modo == 1){
        printf("buscando a lo borracho......."); fflush(stdout);
        npos = search_seq_borracho(mat, n, grupo);
        printf("done\n"); fflush(stdout);
    }
    double t2 = omp_get_wtime();
    printf("Nino encontrado en pos (i,j) --> (%i, %i)\n", npos.first, npos.second);
    printf("Tiempo busqueda: %f secs\n", t2-t1);
    exit(EXIT_SUCCESS);
}
