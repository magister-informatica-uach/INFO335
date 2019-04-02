#ifndef TOOLS_H
#define TOOLS_H

void init_mat_const(int *mat, int n, int c){
    #pragma omp parallel for
    for(int i=0; i<n; ++i){
        for(int j=0; j<n; ++j){
            long index = (long)i*n + (long)j;
            mat[index] = c;
        }
    }
}

void print_mat(int *mat, int n){
    for(int i=0; i<n; ++i){
        for(int j=0; j<n; ++j){
            long index = i*n + j;
            if(mat[index] != 0){
                printf("%i ", mat[index]);
            }
            else{
                printf("* ");
            }
        }
        printf("\n");
    }
}

void print_borracho_mat(int *mat, int n, std::pair<int,int> b){
    long index = b.first * n + b.second;
    mat[index] = 9;
    print_mat(mat, n);
    mat[index] = 0;
}


// busqueda sobrio, una persona busca
std::pair<int, int> search_seq_sobrio(int *mat, int n, int g){
    int seguir=1;
    std::pair<int,int> p(-1,-1);
    #pragma omp parallel shared(seguir, p) 
    {
        int tid = omp_get_thread_num();
        int size = n/omp_get_num_threads();
        int start = size*tid;
        int end = size*tid + size;
        // filas
        for(int i=start; i<end && seguir; i++){
            // columnas
            for(int j=0; j<n && seguir; j++){
                long index = (long)i*n + (long)j;
                if(mat[index] == 1){
                    p = std::pair<int,int>(i, j);
                    #pragma omp critical
                    {
                        seguir = 0;
                    }
                }
            }
        }
    }
    return p;
    /*
    #pragma omp parallel for shared(seguir) schedule(static, chunksize) 
    for(int i=0; i<n; ++i){
        if(seguir){
            break;
        }
        for(int j=0; j<n; ++j){
            if(!seguir){
                break;
            }
            long index = i*n + j;
            if(mat[index] == 1){
                p = std::pair<int,int>(i, j);
                seguir = 0;
            }
        }
    }
    return p;
    */
}

// busqueda sobrio, una persona busca
std::pair<int, int> search_seq_borracho(int *mat, int n, int g){
    int di, dj;
    std::random_device r;
    // asumir borracho en n/2, n/2
    std::pair<int, int> b(n/2, n/2);
    long index = b.first*n + b.second;
    int val = mat[index];
    while(val == 0){
        // seguir caminando mientras no se encuentra
        di = (r() % 3)-1;
        dj = (r() % 3)-1;
        b.first = std::min(std::max(0, b.first+di), n-1);
        b.second = std::min(std::max(0, b.second+dj), n-1);
        index = b.first*n + b.second;
        val = mat[index];
        #ifdef DEBUG
            if(n < 50){
                print_borracho_mat(mat, n, b);
                getchar();
            }
        #endif
    }
    return b;
}
#endif
