#ifndef TOOLS_H
#define TOOLS_H

void init_mat_const(int *mat, int n, int c){
    for(int i=0; i<n; ++i){
        for(int j=0; j<n; ++j){
            long index = i*n + j;
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
    for(int i=0; i<n; ++i){
        for(int j=0; j<n; ++j){
            long index = i*n + j;
            if(mat[index] == 1){
                return std::pair<int,int>(i, j);
            }
        }
    }
    return std::pair<int, int>(-1,-1);
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
