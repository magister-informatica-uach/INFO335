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

void print_mat(int *mat, int n, std::vector<std::pair<int,int>> b){
    for(int i=0; i<b.size(); ++i){
        long index = b[i].first * n + b[i].second;
        mat[index] = 9;
    }
    print_mat(mat, n);
    for(int i=0; i<b.size(); ++i){
        long index = b[i].first * n + b[i].second;
        mat[index] = 0;
    }
}


// busqueda deterministica 
std::pair<int, int> search_det(int *mat, int n, int g){
}

// busqueda no-deterministica
std::pair<int, int> search_nondet(int *mat, int n, int g){
}
#endif
