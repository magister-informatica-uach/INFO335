#ifndef CPURED_H
#define CPURED_H
void gen_random_array(int *a, int n){
    printf("generating array of %i numbers.......", n); fflush(stdout);
    for(int i = 0; i<n; ++i){
        a[i] = rand() % 10;
    }
    printf("done\n"); fflush(stdout);
}

void print_array(int *a, int n, const char *msg){
    if(n > 32){ return; };
    printf("%s: {", msg, n);
    for(int i = 0; i<n-1; ++i){
        printf("%i, ", a[i]);
    }
    printf("%i}\n", a[n-1]);
}

int omp_reduction(int *a, int n, int nt){
    printf("openmp reduction......."); fflush(stdout);
    omp_set_num_threads(nt);
    double t1 = omp_get_wtime();
    int sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for(int i = 0; i<n; ++i){
        sum += a[i];
    }
    double t2 = omp_get_wtime();
    omp_set_num_threads(1);
    printf("done: %f secs\n", t2-t1); fflush(stdout);
    return sum;
}
#endif
