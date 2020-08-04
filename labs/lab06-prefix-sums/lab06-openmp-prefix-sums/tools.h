int log2i(int x){
    int r = 0;
    x >>= 1;
    while(x){
        r++;
        x >>= 1;
    }
    return r;
}

int pow2i(int x){
   return (1 << x); 
}

void printarray(int *a,int n, const char* msg){
	if (n<=32) {
        printf("%s: ",msg);
        for (int i = 0; i <n; ++i){
            printf("%4i ",a[i]);
        }
        printf("\n");
    }
}

void initarray(int *a, int n, const int c){
    #pragma omp parallel for
	for (int i = 0; i < n; ++i){
		a[i] = c*(i+1);
	}
}

void validate(int *s, int *sgold, int n){
	for (int i = 0; i < n; ++i){
		if(s[i] != sgold[i]){
            printf("failed\n");
            printf("Error s[%i](%i)  !=   sgold[%i](%i)\n", i, s[i], i, sgold[i]);
            exit(EXIT_FAILURE); 
        }
	}
}
