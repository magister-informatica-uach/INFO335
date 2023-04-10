// Prefix Sum secuencial
void psums_seq(int *x,int *s,int n){
    if(n == 0){ return; }
    s[0] = x[0];
	for (int i = 1; i < n; ++i){
		s[i]=s[i-1] + x[i];
	}
}

// (1) Prefix Sum recursivo
void psums_rec(int *x,int *s,int n){

}

// (2) Prefix Sum iterativo
void psums_it(int *x,int *s,int n){

}
