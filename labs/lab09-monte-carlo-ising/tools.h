#pragma once

// energy tables
void fill_exptable(real* exptable, const real beta) {
    const int i = omp_get_thread_num();
    if (i == 0) {
        for (int dE = -4; dE <= 4; dE++) {
            exptable[4 + dE] = exp(-beta * 2 * dE);
        }
    }
    return;
}

// print lattice in screen
void print_lattice(int *lattice, const int Lx, const int Ly){
  for (int i = 0; i < Ly; i++){
    for (int j = 0; j < Lx; j++){
	    if( lattice[i*Lx + j] == 1){
		    printf("*");
	    }
	    else{
		    printf(" ");
	    }
    }
    printf("\n");
  }
}

// initialize lattice array to all 1
void initialize_lattice(int* lattice, const int Lx, const int Ly, std::mt19937 &prng) {
    std::uniform_real_distribution<> dist(0.0, 1.0);
    for (int y = 0; y < Ly; y++) {
      for (int x = 0; x < Lx; x++) {
        lattice[x + Lx * y] = dist(prng) > 0.5? 1 : -1;
      }
    }
    return;
}

