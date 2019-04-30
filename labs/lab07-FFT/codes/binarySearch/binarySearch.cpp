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

#define PRINT 0
#define TEST 1
#define CELLS 64	// minimum number of cells to do bs

ulong REPET = 100000;
uint BLK = 32;	// length of the block for sample array (normal distribution) or for increments (uniform distribution)
uint bM; 		// bits for MAX

// Structure with all globals parameters program
typedef struct {
	char prefixResult[300];	// Prefix name of results files in the experiments
	ulong *A;		// original input array
	ulong *X;		// small array for A
	ulong nWX;		// number of Words for X[]

	ulong *PATT;
	ulong n;
	ulong sizeA, sizeX;
	ulong min, MAX;

	bool NORMAL;	// flag probability: 1 = NORMAL, 0 = UNIFORM
	uint sigma;		// for normal distribution probability function

	uint s;
	uint bs;
	uint nc;		// number of cells in COUNT
	uint *COUNT;	// array to store the numbers of items per block
	ulong *POS;		// array to store the position in X where each segment begins
} ParProg;

void genArrays(ParProg *par);
void runBS(ParProg *par);
void runBSSScn(ParProg *par);
bool binarySearch(ParProg *par, ulong x, ulong *idx);
bool scanBSearch(ParProg *par, ulong x, ulong *idx);
void testSearches(ParProg *par);

// 50 /home/hferrada/Dropbox/UACh/teaching/magister/HPC/slides/practicaBSearch/results/ 8 1 10
// 100000000 /home/hferrada/Dropbox/UACh/teaching/magister/HPC/slides/practicaBSearch/results/ 40000 1 10000000
int main(int argc, char** argv){
	ParProg *par = new ParProg();

	if(argc < 5){
		cout << "Execution Error! call: ./bsearch <n> <prefixResult> <BLK> <NORMAL flag> [<sigma>]" << endl;
		exit(EXIT_FAILURE);
	}
	par->n = atoi(argv[1]);
	strcpy(par->prefixResult, "");
	strcpy(par->prefixResult, argv[2]);
	BLK = atoi(argv[3]);
	par->NORMAL = atoi(argv[4]);
	if (par->NORMAL)
		par->sigma = atoi(argv[5]);

	cout << "Parameters..." << endl;
	cout << "n = " << par->n << endl;
	cout << "prefixResult = " << par->prefixResult << endl;
	cout << "NORMAL flag = " << par->NORMAL << endl;
	if (par->NORMAL)
		cout << "sigma = " << par->sigma << endl;

	genArrays(par);
	if (TEST){
		testSearches(par);
		cout << " Test OK !! " << endl;
	}

	runBS(par);
	runBSSScn(par);

	cout << "################## " << endl;
	return 0;
}

// we look for the final position of val in X[1..n]
void heapify(ulong *X, ulong n, ulong val){
	ulong m=2, i=1;

	while(m<n){
		if (m+1 < n && X[m] < X[m+1])
			m++;

		if(val < X[m]){
			X[i] = X[m];
			i = m;
			m <<= 1;
		}else
			break;
	}
	X[i] = val;
}

// heap sort to sort the array
void sortArray(ulong *X, ulong n){
	ulong i,j,k,val;

	// 1.- create the max-heap...
	for(i=2; i<=n; i++){
		val=X[i];
		k=i/2;
		j=i;
		while(k && val>X[k]){
			X[j]=X[k];
			j=k;
			k=k/2;
		}
		X[j]=val;
	}

	if (TEST){
		for (i=n; i>1; i--){
			if (X[i] > X[i/2]){
				cout << "ERROR. X["<<i<<"]=" << X[i] <<" > X[" << i/2 <<"]=" << X[i/2] << endl;
				exit(1);
			}
		}
	}
	/*cout << "H[] =" << endl;
	for (i=1; i<=n; i++)
		cout << X[i] << " ";
	cout << endl;*/

	// 2.- Create the final array...
	k=X[n];
	for(i=n-1; i; i--){
		val=X[1];
		heapify(X,i,k);
		k=X[i];
		X[i]=val;
	}

	if (X[1]>X[2]){
		j=X[1];
		X[1] = X[2];
		X[2] = j;
	}

	if(k<X[1])
		X[0]=k;
	else{
		X[0]=X[1];
		X[1]=k;
	}

	if (TEST){
		for (i=1; i<n; i++){
			if (X[i]<X[i-1]){
				cout << "ERROR. X["<<i<<"]=" << X[i] <<" < X[" << i-1 <<"]=" << X[i-1] << endl;
				exit(1);
			}
		}
	}

}

// generate X nondecreasing array, PATT array for experiments and sample array for bSearch
void genArrays(ParProg *par){
	ulong i, j, k;
	long int num;

	par->sizeA = par->n*sizeof(ulong);	// Original size

	if (par->NORMAL){
		par->A = new ulong[par->n+1];

		default_random_engine generator;
		normal_distribution<double> distribution(4*par->sigma, par->sigma);	// (esperanza, varianza)

	    num = distribution(generator);
	    par->A[0] = par->MAX = num;

	    for (i=1; i<par->n; i++){
	    	num = distribution(generator);
	    	while (num<0)
	    		num = distribution(generator);
	    	par->A[i] = num;
			if (num > (long int)par->MAX)
				par->MAX = num;
	    }
	    par->A[par->n] = par->A[0];
	    sortArray(par->A, par->n);
	}else{
		par->A = new ulong[par->n];
		par->A[0] = rand()%BLK;
		for (i=1; i<par->n; i++)
			par->A[i] = par->A[i-1] + rand()%BLK;
		par->MAX = par->A[par->n-1];
	}

	par->min = par->A[0];
	cout << "min = " << par->min << ", MAX = " << par->MAX << endl;

	// patterns for experiments
	par->PATT = new ulong[REPET];
	k = par->MAX + BLK;
	for (i=0; i<REPET; i++)
		par->PATT[i] = rand()%k;

	// create sample scan structure...
	par->bs = log2(par->MAX/BLK);
	par->s = pow(2,par->bs);
	par->nc = 1 + par->MAX/par->s;
	par->COUNT = new uint[par->nc];
	par->POS = new ulong[par->nc];
	par->POS[0] = 0;
	uint c=0;
	for (i=j=0; i<par->nc; i++){
		k=(i+1)*par->s;
		for (c=0; j<par->n && par->A[j]<k; j++)
			c++;
		par->COUNT[i]=c;
		if ((i+1)<par->nc)
			par->POS[i+1] = par->POS[i]+c;
	}
	cout << "s = " << par->s << endl;
	cout << "bs = " << par->bs << endl;
	cout << "AGV COUNT = " << (float)par->n/(float)par->nc << endl;
	uint aux = par->nc*(sizeof(uint)+sizeof(ulong));
	cout << " Extra size for COUNT[] + POS[] = " << aux/1024.0 << " KiB" << endl;

	// **************************************************************************
	// create X[] array...
	bM = 1+log2(par->MAX);
	par->nWX = (par->n*bM)/(sizeof(ulong)*8);
	if ((par->n*bM)%(sizeof(ulong)*8)>0)
		par->nWX++;

	par->X = new ulong[par->nWX];
	par->sizeX = par->nWX*sizeof(ulong);

	cout << "MAX = " << par->MAX << endl;
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

		/*cout << "PATT[] =" << endl;
		for (i=0; i<REPET; i++)
			cout << par->PATT[i] << " ";
		cout << endl;*/

		cout << "COUNT[0.." << par->nc << "] = ";
		for (i=0; i<par->nc; i++)
			cout << par->COUNT[i] << " ";
		cout << endl;

		cout << "  POS[0.." << par->nc << "] = ";
		for (i=0; i<par->nc; i++)
			cout << par->POS[i] << " ";
		cout << endl;

		c=0;
		for(i=0; c<par->nc; i+=par->COUNT[c], c++);
		if (i!=par->n){
			cout << "ERROR, count cells = " << i << " != n = " << par->n << endl;
			exit(0);
		}
	}
}

void runBS(ParProg *par){
	ulong k, nOcc, pos;
	float avgTime;
	char aFile[400];
	char str[100];
	clock_t t;

	cout << "_________________________________________________" << endl;
	cout << "  Executing " << REPET << " Binary Search on X[] " << endl;

	t = clock();
	for (k=nOcc=0; k<REPET; k++)
		nOcc += binarySearch(par, par->PATT[k], &pos);

	t = clock() - t;
	avgTime = (float)t/CLOCKS_PER_SEC;
	cout << "Average CPU time per execution: " << (avgTime*1000000.0)/REPET << " Microseconds" << endl;
	cout << "nOcc = " << nOcc << endl;

	strcpy(aFile, par->prefixResult);
	strcpy(str, "");
	sprintf(str, "bSearchBLK%d", BLK);
	strcat(aFile, str);
	cout << "Resume File: " << aFile << endl;
	// /home/hferrada/Dropbox/UACh/teaching/magister/practicas/busqueda

	FILE *fp = fopen(aFile, "a+" );
	if (par->NORMAL){
		// [n] [REPET] [nOcc] [avg bs-time/exec] [esperanza] [varianza]
		fprintf(fp, "%ld %ld %ld %f %ld %d\n", par->n, REPET, nOcc, (avgTime*1000000.0)/REPET, par->n/2, par->sigma);
	}else{
		// [n] [REPET] [nOcc] [avg bs-time/exec]
		fprintf(fp, "%ld %ld %ld %f\n", par->n, REPET, nOcc, (avgTime*1000000.0)/REPET);
	}
	fclose(fp);
}

void runBSSScn(ParProg *par){
	ulong k, nOcc, pos;
	float avgTime;
	char aFile[400];
	char str[100];
	clock_t t;

	cout << "_________________________________________________" << endl;
	cout << "  Executing " << REPET << " Binary Search Sample Scan on X[] " << endl;

	t = clock();
	for (k=nOcc=0; k<REPET; k++)
		nOcc += scanBSearch(par, par->PATT[k], &pos);

	t = clock() - t;
	avgTime = (float)t/CLOCKS_PER_SEC;
	cout << "Average CPU time per execution: " << (avgTime*1000000.0)/REPET << " Microseconds" << endl;
	cout << "nOcc = " << nOcc << endl;

	strcpy(aFile, par->prefixResult);
	strcpy(str, "");
	sprintf(str, "bSearchScanBLK%d", BLK);
	strcat(aFile, str);
	cout << "Resume File: " << aFile << endl;
	FILE *fp = fopen(aFile, "a+" );

	if (par->NORMAL){
		// [n] [REPET] [nOcc] [avg bs-time/exec] [esperanza] [varianza]
		fprintf(fp, "%ld %ld %ld %f %ld %d\n", par->n, REPET, nOcc, (avgTime*1000000.0)/REPET, par->n/2, par->sigma);
	}else{
		// [n] [REPET] [nOcc] [avg bs-time/exec]
		fprintf(fp, "%ld %ld %ld %f\n", par->n, REPET, nOcc, (avgTime*1000000.0)/REPET);
	}
	fclose(fp);
}

// binary search for x on X[]
bool binarySearch(ParProg *par, ulong x, ulong *idx){
	ulong *A = par->A;
	if (x < A[0] || x >par->MAX)
		return 0;

	ulong l, r, m;

	l=0;
	r=par->n-1;
	m = r/2;

	while (l<=r){
		if (x==A[m]){
			*idx = m;
			return 1;
		}

		if (x<A[m])
			r=m-1;
		else
			l=m+1;

		m=l+(r-l)/2;
	}
	return 0;
}

bool scanBSearch(ParProg *par, ulong x, ulong *idx){
	ulong *X = par->X;
	if (x < par->min || x >par->MAX)
		return 0;

	uint pos = x>>par->bs;
	ulong c=par->COUNT[pos];
	ulong m, xm, l=par->POS[pos];

	if (c > CELLS){
		// Binary searching in the X-segment...
		ulong r;

		r=l+c-1;
		m=(l+r)>>1;
		xm = getNum64(X, m*bM, bM);

		while (l<=r){
			if (x<xm)
				r=m-1;
			else{
				if (x==xm){
					*idx = m;
					return 1;
				}
				l=m+1;
			}
			m=(l+r)>>1;
			xm=getNum64(X, m*bM, bM);
		}
	}else{
		// Scanning a X-segment of maximum c cells...
		//for (m=l*bM, xm=getNum64(X, m, bM); xm<x; m+=bM, xm=getNum64(X, m, bM))
		m=l*bM;
		xm=getNum64(X, m, bM);
		while (xm<x){
			m+=bM;
			xm=getNum64(X, m, bM);
		}

		if (xm==x){
			*idx = m/bM;
			return 1;
		}
	}

	return 0;
}


void testSearches(ParProg *par){
	ulong k, p1=par->n+1, p2=par->n+2;
	bool found;

	for (k=0; k<REPET; k++){
		p1=par->n+1;
		p2=0;

		found = binarySearch(par, par->PATT[k], &p1);
		if (scanBSearch(par, par->PATT[k], &p2))
			found = 1;

		if (found && par->A[p1] != par->A[p2] ){
			cout << "ERROR. patt[" <<k<< "] = " << par->PATT[k] << " binarySearch().pos = " << p1 << " != scanBSearch().pos = " << p2 << endl;
			exit(1);
		}
	}
}
