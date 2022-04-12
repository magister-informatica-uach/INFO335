#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// --> 0 --> 1 --> 2 --> 3 --> vuelta...


void llenar_matriz(int *m, int n){
	for(int i=0; i<n*n; ++i){
		m[i] = i;
	}
}

void llenar_matriz_zero(int *m, int n){
	for(int i=0; i<n*n; ++i){
		m[i] = 0;
	}
}

void imprimir_matriz(int *m, int n, int rank){
	printf("Proceso %i:\n", rank);
	for(int i=0; i<n; ++i){
		for(int j=0; j<n; ++j){
			printf("%i ", m[i*n + j]);
		}
		printf("\n");
	}
}

int main(int argc, char** argv) {
    // PREAMBULO 
    MPI_Init(NULL, NULL);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);


    // ALGORITMO
    if(rank == 0){
        char historia[1024] = "habia una vez...0";
        // MPI_Send(punteroBuffer, bytes, tipo_MPI, rank_destino, 0, scope_comunicacion);
        MPI_Send(historia, 1024, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(historia, 1024, MPI_CHAR, size-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        char str[10];
        sprintf(str, "-->%d", rank);
        strcat(historia, str);
        printf("PROCESO %i: %s\n", rank, historia);
    }
    else{
        char msg[1024];
        char str[10];
        sprintf(str, "-->rank %d", rank);
        MPI_Recv(msg, 1024, MPI_CHAR, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        strcat(msg, str);
        MPI_Send(msg, 1024, MPI_CHAR, (rank+1) % size, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
}


