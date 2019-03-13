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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define real double

// random number uniformly distributed in [0,1)
inline real rand_uniform(unsigned* rng) {
  *rng = 1664525 * (*rng) + 1013904223;
  return (2.32830643653869629E-10 * (*rng));
} 

// seed rng states
void seed_rng_states(unsigned* rngs, const unsigned seed) {
  #pragma omp parallel
  {
    const int i = omp_get_thread_num();
    unsigned state = seed + 11 * i;
    rngs[i] = (unsigned)(1e6 * rand_uniform(&state));
  }
  return;
}

// initialize spin array to all 1
void initialize_spins(int* spin, const int L) {
  #pragma omp parallel
  {
    #pragma omp for
    for (int y = 0; y < L; y++) {
      for (int x = 0; x < L; x++) {
        spin[x + L * y] = 1;
      }
    }
  }
  return;
}

// energy tables
void fill_exptable(real* exptable, const real beta) {
  #pragma omp parallel
  {
    const int i = omp_get_thread_num();
    if (i == 0) {
      for (int dE = -4; dE <= 4; dE++) {
        exptable[4 + dE] = exp(-beta * 2 * dE);
      }
    }
  }
  return;
}

// metropolis sublattice update
void update_spins_sublattice(const int sublattice, unsigned* rngs, 
                             int* spin, const int L, 
                             const real* exptable) {
  #pragma omp parallel
  {
    unsigned rng_copy = rngs[omp_get_thread_num()];
    #pragma omp for
    for (int y = 0; y < L; y++) {
      for (int x = 0; x < L; x++) {
        if (sublattice == ((x + y) & 1)) {
          int sum = 0;
          sum += spin[(x == (L - 1) ? 0 : x + 1) + L * (y)];
          sum += spin[(x == 0 ? (L - 1) : x - 1) + L * (y)];
          sum += spin[x + L * (y == (L - 1) ? 0 : y + 1)];
          sum += spin[x + L * (y == 0 ? (L - 1) : y - 1)];
          int dE = spin[x + L * y] * sum;
          if (exptable[4 + dE] > rand_uniform(&rng_copy))
            spin[x + L * y] = -spin[x + L * y];
        }
      }
    }
    rngs[omp_get_thread_num()] = rng_copy;
  }
  return;
}

// measure magnetization
real measure_spins(int* spin, const int L) {
  real sum = 0.0;
  #pragma omp parallel for reduction(+:sum)
  for (int y = 0; y < L; y++) {
    for (int x = 0; x < L; x++) {
      sum += spin[x + L * y];
    }
  }
  return fabs(sum / (1.0 * L * L));
}

// sample
void average_measure(unsigned* rngs, int* spin, const int L, 
                     const real* exptable, const int steps, 
                     real* m_mean, real* m_error) {
  real m[10] = {0};
  for (int w = 0; w < 10; w++) {
    real sum = 0.0;
    for (int t = 0; t < steps; t++) {
      update_spins_sublattice(0, rngs, spin, L, exptable);
      update_spins_sublattice(1, rngs, spin, L, exptable);
      sum += measure_spins(spin, L);
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

void print_lattice(int *spin, const int L){
}

int main(int argc, char** argv) {
    if(argc != 3){
        fprintf(stderr, "run as ./prog L steps\n");
        exit(1);
    }
    // parameters
    const int L = atoi(argv[1]);
    const int steps  = atoi(argv[2]);
    int* spin;
    // measurements
    // set rng streams
    unsigned* rngs;
    rngs = (unsigned*)malloc(sizeof(unsigned) * omp_get_max_threads());
    seed_rng_states(rngs, 123456789);
    // initialize spins
    spin = (int*)malloc(sizeof(int) * L * L);    
    initialize_spins(spin, L);
    printf("Initial configuration\n");
    print_lattice(spin, L);
    getchar();
    // initialize energy tables
    real* exptable;
    exptable = (real*)malloc(sizeof(real) * 9);    
    real kT = 2.0 / log(1 + sqrt(2));
    fill_exptable(exptable, 1.0 / kT);
    // equilibriate
    int steps_equilibriate = steps;
    for (int t = 0; t < steps_equilibriate; t++) {
      printf("t=%i:\n", t);
      update_spins_sublattice(0, rngs, spin, L, exptable);
      update_spins_sublattice(1, rngs, spin, L, exptable);  
      print_lattice(spin, L);
      getchar();
    }
    const int steps_sample = steps;
    real mean, error;
    real start = omp_get_wtime();    
    average_measure(rngs, spin, L, exptable, steps_sample, &mean, &error);
    real end = omp_get_wtime(); 
    // print results
    printf("final lattice configuration\n");
    print_lattice(spin, L);
    // free arrays
    free(exptable);
    free(spin);
    free(rngs);
    return 0;
}
