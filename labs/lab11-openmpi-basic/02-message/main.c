#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>


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
    if(argc != 2){
	    fprintf(stderr, "run as mpirun -np X prog n\n");
	    exit(EXIT_FAILURE);
    }
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);
    // Get the number of processes
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // Get the rank of the process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    // (1) pasar una matriz de nxn de proceso 0 a 1, el 1 la llena, y 
    // la envia de vuelta al 0
    if(rank == 0){
    	int n = atoi(argv[1]);
	// a) master crea la matriz
	int *mat = (int*)malloc(sizeof(int)*n*n);
	llenar_matriz_zero(mat, n);
	imprimir_matriz(mat, n, rank);
	// b) master manda n
	MPI_Send(&n, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
	// c) master manda la matriz a esclavo 
	MPI_Send(mat, n*n, MPI_INT, 1, 1, MPI_COMM_WORLD);
	// d) master espera la matriz con valores
	MPI_Recv(mat, n*n, MPI_INT, 1, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	imprimir_matriz(mat, n, rank);
    }
    else{
    	//printf("Hello world from processor %s, rank %d out of %d processors\n", processor_name, rank, size);
	int n=-1;
	// recibir el n
	MPI_Recv(&n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	int *slavemat = (int*)malloc(sizeof(int)*n*n);
	// recibir la matriz	
	MPI_Recv(slavemat, n*n, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	llenar_matriz(slavemat, n);
	// mandar matriz
	MPI_Send(slavemat, n*n, MPI_INT, 0, 5, MPI_COMM_WORLD);
    }
    // Finalize the MPI environment.
    MPI_Finalize();
}

