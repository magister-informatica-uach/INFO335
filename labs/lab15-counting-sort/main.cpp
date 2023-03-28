// C++ Program for counting sort
#include <bits/stdc++.h>
#include <string.h>
#include <cstdio>
using namespace std;
 
void print_array(int *arr, int n, const char *msg);
// The main function that sort
// the given string arr[] in
// alphabetical order
int* countSort(int *arr, int n, int k){
    // The output array
    // that will have sorted arr
    int *output = (int*)malloc(sizeof(int)*n);
 
    // Create a count array to store count of individual
    // ints and initialize count array as 0
    int count[k + 1];
    memset(count, 0, sizeof(count));
 
    // Store count of each int
    for (int i=0; i < n; ++i){
        //printf("visiting arr[%i] = %i\n", i, arr[i]);
        ++count[arr[i]];
    }
 
    // Change count[i] so that count[i] now contains actual
    // position of this int in output array
    for (int i = 1; i <= k; ++i)
        count[i] += count[i - 1];

    //print_array(count, k+1, "count array");
 
    // Build the output int array
    for (int i=0; i < n; ++i){
        output[count[arr[i]] - 1] = arr[i];
        --count[arr[i]];
    }
    //print_array(output, n, "output array");
    return output;
}
 
// Driver  code
int main(int argc, char **argv){
    if(argc != 4){
        fprintf(stderr, "run as ./prog seed n k\n");
        exit(EXIT_FAILURE);
    }
    int seed = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);
    printf("seed %i  n=%i  k=%i\n", seed, n, k);
    int arr[n];
    srand(seed);
    for(int i=0; i<n; ++i){
        arr[i] = rand() % k;
    }
    print_array(arr, n, "Unsorted");
    int *out = countSort(arr, n, k);
    print_array(out, n, "Sorted");
    return 0;
}

void print_array(int *arr, int n, const char *msg){
    printf("%s[n=%i]:\n", msg, n);
    for(int i=0; i<n; ++i){
        printf("%i, ", arr[i]);
    }
    printf("\n\n");
}
// This code is contributed by rathbhupendrai
