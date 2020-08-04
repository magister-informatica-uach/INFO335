#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define REPEATS 1.0

#include "tools.h"
#include "algs.h"

const char* algs[2] = {"Recursive", "Iterative"};

int main(int argc, char **argv){
	double t1,t2;
	if(argc != 4){
		fprintf(stderr, "argc = %i\n", argc); fflush(stderr);
        fprintf(stderr, "run as ./prog n alg nt\n"); fflush(stderr);
		exit(EXIT_FAILURE);
	}
	int n=atoi(argv[1]);
    int alg=atoi(argv[2]);
	int nt=atoi(argv[3]);
	int *s= new int[n];
	int *sgold= new int[n];
	int *x= new int[n];
	omp_set_num_threads(nt);
    printf("Using %i OpenMP threads\n", nt);




    // (1) INIT DATA
    printf("Init data.............................."); fflush(stdout);
	t1=omp_get_wtime();
    initarray(x, n, 1.0f);
	t2=omp_get_wtime();
    printf("done: %f secs\n", t2-t1);
	printarray(x,n,"x: ");




    // (2) PARALLEL PREFIX SUM
    printf("\nParallel Prefix Sum [%s].......", algs[alg]); fflush(stdout);
	t1=omp_get_wtime();
    if(alg==0){
	    psums_rec(x,s,n);
    }
    else{
	    psums_it(x,s,n);
    }
	t2=omp_get_wtime();
    printf("done: %f secs\n",(t2-t1));
	printarray(s,n,"s: ");





    // (3) [GOLD] SEQUENTIAL PREFIX SUMS
    printf("\nSequential Prefix Sum................."); fflush(stdout);
	t1=omp_get_wtime();
    psums_seq(x, sgold, n);
	t2=omp_get_wtime();
    printf("done: ");
	printf("%f secs\n",(t2-t1));
	printarray(sgold,n,"s: ");
    printf("\nValidating............................."); fflush(stdout);





    // (4) VALIDATE PARALLEL RESULT WITH GOLD
    validate(s, sgold, n);
    printf("pass\n");
	return 0;
}
