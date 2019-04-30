#include <iostream>
#include <omp.h>
#include <vector>
#include <random>
using namespace std;

int main() {
    random_device r;
    std::vector<std::default_random_engine> generators;
    for (int i = 0, N = omp_get_max_threads(); i < N; ++i) {
        generators.emplace_back(default_random_engine(r()));
    }

    int N = 1000;
    vector<int> v(N);

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        // Get the generator based on thread id
        default_random_engine& engine = generators[omp_get_thread_num()];
        // Perform heavy calculations
        uniform_int_distribution<int> uniform_dist(1, 100);
        v[i] = uniform_dist(engine); // I assume this is thread unsafe
    }
    return 0;
}
