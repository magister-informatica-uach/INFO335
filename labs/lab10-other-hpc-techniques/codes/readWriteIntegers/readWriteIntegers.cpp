#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <random>
#include <bits/random.h>
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */
#include "include/BasicCDS.h"

using namespace std;
using namespace cds;

#define PRINT 1
#define TEST 1

uint bM; 		// bits for MAX

// Structure with all globals parameters program
typedef struct {
	ulong *A;
	ulong *X;
	ulong n;
	ulong MAX;

	ulong sizeA, sizeX;
	ulong nWX;		// number of Wrds for X[]
} ParProg;

void genArrays(ParProg *par);

// 100 1 5
int main(int argc, char** argv){
	ParProg *par = new ParProg();

	if(argc != 3){
		cout << "Execution Error! call: ./readWriteInt <n> <MAX>" << endl;
		exit(EXIT_FAILURE);
	}
	par->n = atoi(argv[1]);
	par->MAX = atoi(argv[2]);

	cout << "Parameters..." << endl;
	cout << "n = " << par->n << endl;
	cout << "MAX = " << par->MAX << endl;

	genArrays(par);

	cout << "################## " << endl;
	return 0;
}

// generate X nondecreasing array, PATT array for experiments and sample array for bSearch
void genArrays(ParProg *par){
	ulong i, j, k;

	par->A = new ulong[par->n];
	par->sizeA = par->n*sizeof(ulong);
	for (i=0; i<par->n; i++)
		par->A[i] = rand()%par->MAX;

	// **************************************************************************
	// create X[] array...
	bM = 1+log2(par->MAX);
	par->nWX = (par->n*bM)/(sizeof(ulong)*8);
	if ((par->n*bM)%(sizeof(ulong)*8)>0)
		par->nWX++;

	par->X = new ulong[par->nWX];
	par->sizeX = par->nWX*sizeof(ulong);

	cout << "bM = " << bM << endl;
	cout << " size for A[] = " << par->sizeA/(1024.0*1024.0) << " MiB" << endl;
	cout << " size for X[] = " << par->sizeX/(1024.0*1024.0) << " MiB" << endl;

	for (i=j=0; i<par->n; i++, j+=bM)
		setNum64(par->X, j, bM, par->A[i]);

	// **************************************************************************

	if (PRINT){
		cout << "A[] = ";
		for (i=0; i<par->n; i++)
			cout << par->A[i] << " ";
		cout << endl;

		cout << "X[] = ";
		for (i=j=0; i<par->n; i++, j+=bM)
			cout << getNum64(par->X, j, bM) << " ";
		cout << endl;

		/*cout << "X[] = ";
		for (i=0; i<par->nWX; i++){
			printBitsUlong(par->X[i]);
			cout << " - ";
		}
		cout << endl;*/
	}

	if(TEST){
		for (i=j=0; i<par->n; i++, j+=bM){
			k = getNum64(par->X, j, bM);
			if (k != par->A[i]){
				cout << "ERROR, A["<<i<<"] = " << par->A[i] << " != X["<<i<< "] = " << k << endl;
				exit(0);
			}
		}
		cout << "Test OK !!" << endl;
	}
}
