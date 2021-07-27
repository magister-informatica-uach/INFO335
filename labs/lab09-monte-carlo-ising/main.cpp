// square lattice Ising model with checkerboard metropolis update
// parallelization using openmp pragmas

// mic: 
// $ icc -std=c99 -mmic -openmp -fno-inline -O3 ising_omp.c
// $ export KMP_AFFINITY=balanced && export OMP_NUM_THREADS=240
// $ nohup ./a.out 10 100 8 128 test.dat &

// gcc:
// $ gcc -std=c99 -Wall -O3 -fopenmp ising_omp.c -lm
// $ export OMP_NUM_THREADS=2
// $ nohup ./a.out 100 1000 8 128 1 test.dat &
#define real double

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <random>
#include "tools.h"


// metropolis dir update
void update_lattice_dir(const int dir, std::mt19937 &prng, int* lattice, const int Lx, const int Ly, const real* exptable) {
    std::uniform_real_distribution<> dist(0.0, 1.0);
    for (int y = 0; y < Ly; y++) {
        for (int x = 0; x < Lx; x++) {
            if (dir == ((x + y) & 1)) {
                int sum = 0;
                sum += lattice[(x == (Lx - 1) ? 0 : x + 1) + Lx * (y)];
                sum += lattice[(x == 0 ? (Lx - 1) : x - 1) + Lx * (y)];
                sum += lattice[x + Lx * (y == (Ly - 1) ? 0 : y + 1)];
                sum += lattice[x + Lx * (y == 0 ? (Ly - 1) : y - 1)];
                int dE = lattice[x + Lx * y] * sum;
                real randval = dist(prng);
                if (exptable[4 + dE] > randval){
                    lattice[x + Lx * y] = -lattice[x + Lx * y];
                }
            }
        }
    }
    return;
}

// measure magnetization
real measure_lattice(int* lattice, const int Lx, const int Ly){
    real sum = 0.0;
    for (int y = 0; y < Ly; y++) {
        for (int x = 0; x < Lx; x++) {
            sum += lattice[x + Lx * y];
        }
    }
    return fabs(sum / (1.0 * Lx * Ly));
}

// sample
void average_measure(std::mt19937 &prng, int* lattice, const int Lx, const int Ly, const real* exptable, const int steps, real* m_mean, real* m_error) {
  real m[10] = {0};
  for (int w = 0; w < 10; w++) {
    real sum = 0.0;
    for (int t = 0; t < steps; t++) {
      update_lattice_dir(0, prng, lattice, Lx, Ly, exptable);
      update_lattice_dir(1, prng, lattice, Lx, Ly, exptable);
      sum += measure_lattice(lattice, Lx, Ly);
    }
    m[w] = sum / (1.0 * steps);
  }
  // mean
  real mean = 0.0;
  for (int w = 0; w < 10; w++) {
    mean += m[w];
  }
  mean /= 10.0;
  *m_mean = mean;
  // error
  real error = 0.0;
  for (int w = 0; w < 10; w++) {
    error += (m[w] - mean) * (m[w] - mean);
  }
  error /= (9.0);
  *m_error = sqrt(error);
}

int main(int argc, char** argv) {
    if(argc != 4){
        fprintf(stderr, "run as ./prog L steps temp\n");
        exit(1);
    }
    // parameters
    const int L = atoi(argv[1]);
    const int Lx = L;
    const int Ly = L/2;
    const int steps  = atoi(argv[2]);
    const double temp = atoi(argv[3]);
    int* lattice;
    // measurements
    // set rng streams
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 prng(rd()); //Standard mersenne_twister_engine seeded with rd()


    // (1) initialize lattice
    lattice = new int[(Lx)*(Ly)];
    initialize_lattice(lattice, Lx, Ly, prng);
    printf("Initial configuration\n");
    if(Lx <= 256){
        print_lattice(lattice, Lx, Ly);
        getchar();
    }



    // (2) initialize energy tables
    real* exptable;
    exptable = (real*)malloc(sizeof(real) * 9);    
    real kT = temp;
    fill_exptable(exptable, 1.0 / kT);



    // (3) equilibriate [Paralelizar update_lattice_dir]
    int steps_equilibriate = steps;
    for (int t = 0; t < steps_equilibriate; t++) {
      printf("t=%i:\n", t);
      update_lattice_dir(0, prng, lattice, Lx, Ly, exptable);
      update_lattice_dir(1, prng, lattice, Lx, Ly, exptable);  
      if(Lx <= 256){
          print_lattice(lattice, Lx, Ly);
          printf("\n\n\n\n");
          getchar();
      }
    }
    const int steps_sample = steps;
    real mean, error;




    // (4) physical measures
    real start = omp_get_wtime();    
    average_measure(prng, lattice, Lx, Ly, exptable, steps_sample, &mean, &error);
    real end = omp_get_wtime(); 




    // (5) print results
    printf("final lattice configuration\n");
    if(L <= 256){
        print_lattice(lattice, Lx, Ly);
    }
    // free arrays
    free(exptable);
    free(lattice);
    return 0;
}
