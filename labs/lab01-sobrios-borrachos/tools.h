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

void print_borrachos_mat(int *mat, int n, std::vector<std::pair<int,int>> b){
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
    std::random_device r;
    std::vector<std::default_random_engine> generators;
    for (int i = 0, N = g; i < N; ++i){
        //printf("creando generador %i...\n", i);
        generators.emplace_back(std::default_random_engine(r()));
    }
    // asumir borracho en n/2, n/2
    std::pair<int, int> p(-1, -1);
    int seguir = 1;
    std::vector<std::pair<int, int>> b;
    b.resize(g);
    #pragma omp parallel shared(p, seguir, b)
    {
        int di, dj;
        int tid = omp_get_thread_num();
        std::default_random_engine& engine = generators[omp_get_thread_num()];
        std::uniform_int_distribution<int> uniform_dist(-1, 1);
        std::uniform_int_distribution<int> ud2(0, n-1);
        b[tid] = std::pair<int, int>(n/2, n/2);
        long index = b[tid].first*n + b[tid].second;
        int val = mat[index];
        while(val == 0 && seguir){
            // Perform heavy calculations
            di = uniform_dist(engine); // I assume this is thread unsafe
            dj = uniform_dist(engine); // I assume this is thread unsafe

            // seguir caminando mientras no se encuentra
            //di = (r() % 3)-1;
            //dj = (r() % 3)-1;
            b[tid].first = std::min(std::max(0, b[tid].first+di), n-1);
            b[tid].second = std::min(std::max(0, b[tid].second+dj), n-1);
            index = b[tid].first*n + b[tid].second;
            val = mat[index];
            
            #pragma omp barrier
            if(tid == 0){
                #ifdef DEBUG
                    if(n < 50){
                        print_borrachos_mat(mat, n, b);
                        printf("hoa\n");
                        getchar();
                    }
                #endif
            }
            #pragma omp barrier
            
        }
        if(val != 0){
            #pragma omp critical
            {
                seguir = 0;
            }
            p = b[tid];
        }
    }
    return p;
}
#endif
