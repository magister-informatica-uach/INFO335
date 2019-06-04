#include <iostream>
#include <cstdio>
#include <omp.h>

void printarray(int *a, int n, const char *msg){
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

void pointer_jumping(int *p, int n){
	#pragma omp parallel
	{
		int c = 0;
		int tid = omp_get_thread_num();
		while(p[tid] != p[ p[tid] ]){
			p[tid] = p[ p[tid] ]; 
			printf("thread %i  it %i\n", tid, c++);
			//#pragma omp barrier
		}
	}
}
int main(int argc, char **argv){
	if(argc != 3){
		fprintf(stderr, "./prog n nt\n");
		exit(EXIT_FAILURE);
	}
	int n = atoi(argv[1]);
	int nt = atoi(argv[2]);
	int *p = new int[n];
	for(int i=0; i<n-1; ++i){
		p[i] = i+1;
	}
	p[n-1] = n-1;
	printarray(p, n, "arreglo original");
	omp_set_num_threads(nt);
	pointer_jumping(p, n);
	printarray(p, n, "solucion");
	return 0;
}
