/* Análise utilizando perf executado em Intel Core i3-8130u, 2 Cores 4 Threads

Tentativa de paralelização sem a transposta, nota-se elevado cache miss e aumento do tempo em relação ao serial (cerca de 8 segundos no serial).

./perf stat --repeat 2 -e cycles:u,instructions:u,cache-misses:u,L1-dcache-loads:u,L1-dcache-load-misses:u,L1-dcache-stores,dTLB-loads,dTLB-load-misses build/parallel tests/5.in 

tempo: 12.958557 
tempo: 12.829128 

Performance counter stats for 'build/parallel tests/5.in' (2 runs): 

      163418277843      cycles:u                                                      ( +-  1.80% )  (62.49%) 

       30648121306      instructions:u            #    0.19  insn per cycle           ( +-  0.09% )  (75.01%) 

         316156077      cache-misses:u                                                ( +-  0.18% )  (75.05%) 

        6802671740      L1-dcache-loads                                               ( +-  0.13% )  (75.03%) 

        4874789132      L1-dcache-load-misses     #   71.75% of all L1-dcache accesses  ( +-  0.09% )  (75.02%) 

          57196846      L1-dcache-stores                                              ( +-  0.77% )  (75.07%) 

        6809225160      dTLB-loads                                                    ( +-  0.12% )  (49.94%) 

        3117029186      dTLB-load-misses          #   45.83% of all dTLB cache accesses  ( +-  2.06% )  (49.94%) 

  

           12.9775 +- 0.0645 seconds time elapsed  ( +-  0.50% ) 


Com a matriz transposta, nota-se a diminuição de cache miss e melhora muito significativa do tempo de execução.

./perf stat --repeat 2 -e cycles:u,instructions:u,cache-misses:u,L1-dcache-loads:u,L1-dcache-load-misses:u,L1-dcache-stores,dTLB-loads,dTLB-load-misses build/parallel tests/5.in 

tempo: 1.120696 
tempo: 1.161120 

Performance counter stats for 'build/parallel tests/5.in' (2 runs): 

       14751365674      cycles:u                                                      ( +-  0.09% )  (62.53%) 

       27406240248      instructions:u            #    1.86  insn per cycle           ( +-  0.06% )  (74.90%) 

         170441566      cache-misses:u                                                ( +-  5.77% )  (75.12%) 

        6828245621      L1-dcache-loads                                               ( +-  0.10% )  (74.88%) 

         188913339      L1-dcache-load-misses     #    2.77% of all L1-dcache accesses  ( +-  1.16% )  (75.23%) 

          54802750      L1-dcache-stores                                              ( +-  1.51% )  (75.34%) 

        6864069782      dTLB-loads                                                    ( +-  0.16% )  (49.66%) 

           3360910      dTLB-load-misses          #    0.05% of all dTLB cache accesses  ( +-  0.07% )  (50.11%) 

  

            1.2314 +- 0.0133 seconds time elapsed  ( +-  1.08% ) 
*/

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <omp.h>

// Initialize matrices
void initialize_matrices(float *a, float *b, float *c, unsigned size,
                         unsigned seed) {
  srand(seed);
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      a[i * size + j] = rand() % 10;
      b[i * size + j] = rand() % 10;
      c[i * size + j] = 0.0f;
    }
  }
}

// Parallelize this function using OpenMP
void multiply(float *a, float *b, float *c, unsigned size) {
  float *b_transposed = (float *)malloc(sizeof(float) * size * size);
  
  // Transposta para melhorar eficiência dos cache misses.
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      b_transposed[i * size + j] = b[j * size + i];
    }
  }
  
  // Paraleliza loops
  #pragma omp parallel for
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      float sum = 0.0;
      for (int k = 0; k < size; ++k) {
        sum = sum + a[i * size + k] * b_transposed[j * size + k];
      }
      c[i * size + j] = sum;
    }
  }
  
  free(b_transposed);
}

// Output matrix to stdout
void print_matrix(float *c, unsigned size) {
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      printf(" %5.1f", c[i * size + j]);
    }
    printf("\n");
  }
}

int main(int argc, char *argv[]) {
  float *a, *b, *c;
  unsigned seed, size;
  double t;
  FILE *input;

  if (argc < 2) {
    fprintf(stderr, "Error: missing path to input file\n");
    return 1;
  }

  if ((input = fopen(argv[1], "r")) == NULL) {
    fprintf(stderr, "Error: could not open file\n");
    return 1;
  }

  // Read inputs
  fscanf(input, "%u", &size);
  fscanf(input, "%u", &seed);

  // Do not change this line
  omp_set_num_threads(4);

  // Allocate matrices
  a = (float *)malloc(sizeof(float) * size * size);
  b = (float *)malloc(sizeof(float) * size * size);
  c = (float *)malloc(sizeof(float) * size * size);

  // initialize_matrices with random data
  initialize_matrices(a, b, c, size, seed);

  // Multiply matrices
  t = omp_get_wtime();
  multiply(a, b, c, size);
  t = omp_get_wtime() - t;

  // Show result
  print_matrix(c, size);

  // Output elapsed time
  fprintf(stderr, "%lf\n", t);

  // Release memory
  free(a);
  free(b);
  free(c);

  return EXIT_SUCCESS;
}
