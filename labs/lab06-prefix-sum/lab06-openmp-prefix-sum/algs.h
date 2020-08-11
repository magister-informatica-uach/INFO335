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
    // i) caso de termino
    if(n == 1){
        s[0] = x[0];
        return;
    }
    int nh = n >> 1;
    // ii) FASE 1: trabajo en el paso
    int *y = new int[nh];
    int *z = new int[nh];
    #pragma omp parallel for shared(nh, x, y)
    for(int i=0; i<nh; ++i){
        y[i] = x[i << 1] + x[(i << 1) + 1];
    }
    // iii) llamada recursiva
    psums_rec(y, z, nh);
    // iv) recolectar resultados z
    #pragma omp parallel for shared(nh, s, x, z)
    for(int i=0; i<n; ++i){
        // i impar
        if( (i & 1) != 0 ){
           s[i] = z[i >> 1]; 
        }
        // i == 0
        else if(i == 0){
            s[0] = x[0];
        }
        // i impar y i > 0
        else{
            s[i] = z[(i-1) >> 1] + x[i]; 
        }
    }
    free(y);
    free(z);
}

// (2) Prefix Sum iterativo
void psums_it(int *x,int *s,int n){

}
