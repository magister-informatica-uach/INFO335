#include <cstdio>
#include <cstdlib>
#include <random>
#include <omp.h>

void random_init(int *a, int n, int seed){
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dist(1, 100);
    for(int i=0; i<n; ++i){
        a[i] = dist(gen);
        //a[i] = rand();
    }
}

void print_array(int *a, int n, const char *msg){
    if(n >= 30){return;}
    printf("%s:\n", msg);
    for(int i=0; i<n; ++i){
        printf("%i, ", a[i]);
    }
}

int suma(int *a, int n){
    int res = 0;
    for(int i=0; i<n; ++i){
        res += a[i]; 
    }
    return res;
}

int main(int argc, char **argv){
    // 1) variables y argumentos
    //  arg0  arg1 arg2
    // ./prog <n>  <seed>
    double t1, t2;
    if(argc != 3){
        fprintf(stderr, "ejecutar como ./prog <n> <seed>\n");
        exit(EXIT_FAILURE);
    }
    int n = atoi(argv[1]);
    int seed = atoi(argv[2]);


    // 2) alocar memoria e inicializar datos
    printf("inicializando %i datos random (seed=%i)....", n, seed); fflush(stdout);
    // MARCA TIEMPO 1 
    t1 = omp_get_wtime();
    int *array = (int*)malloc(n*sizeof(int));
    random_init(array, n, seed);
    print_array(array, n, "arreglo inicializado");
    // MARCA TIEMPO 2
    t2 = omp_get_wtime();
    printf("done: %f secs\n", t2-t1); fflush(stdout);

    // 3) calcular la suma
    printf("suma...."); fflush(stdout);
    t1 = omp_get_wtime();
    int result = suma(array, n);
    t2 = omp_get_wtime();
    printf("done: %f secs\n", t2-t1); fflush(stdout);

    // 4) mostrar resultado
    printf("SUMA = %i\n", result);

    // 5) liberar 
    free(array);
}
