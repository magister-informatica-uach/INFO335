#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
void printarray(float *a,int n, const char* msg){
	printf("%s\n",msg);
	for (int i = 0; i <n; ++i)
	{
		printf("%f\n",a[i]);
	}
	printf("\n");
}


void prefix(float *x,float *s,int n){
}


int main(int argc, char const *argv[]){
	double t1,t2;
	srand(time(NULL));
	if(argc != 3){
		fprintf(stderr, "run as ./prog n nt\n");
		exit(EXIT_FAILURE);
	}
	int n=atoi(argv[1]);
	int ntt=atoi(argv[2]);
	float *s=(float *)malloc(sizeof(float)*n);
	float *x=(float *)malloc(sizeof(float)*n);
	for (int i = 0; i < n; ++i)
	{
		x[i]=1.0*(i+1);
	}
	omp_set_num_threads(ntt);
	t1=omp_get_wtime();
	prefix(x,s,n);
	t2=omp_get_wtime();
	if (n<=128) {
		printarray(s,n,"s: ");
	}
	printf("prefix sum time: %f [s]\n",t2-t1 );
	
	
	return 0;
}
