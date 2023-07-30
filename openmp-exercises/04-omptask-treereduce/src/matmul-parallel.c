/*
  Perfilamento em um Intel Core i3-8130u, 2 Cores 4 Threads
  
  Execução serial:
  Tempo: 9.195305 

Performance counter stats for 'build/serial tests/5.in' (2 runs): 

       44973944834      cycles:u                                                      ( +-  0.53% )  (62.43%) 

      123872758189      instructions:u            #    2.77  insn per cycle           ( +-  0.07% )  (74.95%) 

          25828227      cache-misses:u                                                ( +-  1.04% )  (74.97%) 

       43286923067      L1-dcache-loads                                               ( +-  0.24% )  (74.99%) 

          70192817      L1-dcache-load-misses     #    0.16% of all L1-dcache accesses  ( +-  5.69% )  (74.99%) 

       15742015475      L1-dcache-stores                                              ( +-  0.20% )  (75.05%) 

       43476287474      dTLB-loads                                                    ( +-  0.26% )  (50.04%) 

           2487796      dTLB-load-misses          #    0.01% of all dTLB cache accesses  ( +- 11.32% )  (49.96%) 

  

           14.1647 +- 0.0566 seconds time elapsed  ( +-  0.40% ) 



Execução paralela:
Tempo: 5.403181 

Performance counter stats for 'build/parallel tests/5.in' (2 runs): 

  

       67113690617      cycles:u                                                      ( +-  0.24% )  (62.62%) 

      123120698979      instructions:u            #    1.83  insn per cycle           ( +-  0.04% )  (75.05%) 

          39095785      cache-misses:u                                                ( +-  7.38% )  (74.96%) 

       42961188214      L1-dcache-loads                                               ( +-  0.07% )  (74.72%) 

         190822442      L1-dcache-load-misses     #    0.44% of all L1-dcache accesses  ( +-  5.99% )  (74.83%) 

       15877488928      L1-dcache-stores                                              ( +-  0.21% )  (75.04%) 

       43512112872      dTLB-loads                                                    ( +-  0.26% )  (50.22%) 

           7049281      dTLB-load-misses          #    0.02% of all dTLB cache accesses  ( +- 10.89% )  (50.25%) 

  

            10.235 +- 0.251 seconds time elapsed  ( +-  2.45% ) 

*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define SEED 123

void free_matrix(int **m, int size) {
  for (int i = 0; i < size; i++)
    free(m[i]);
  free(m);
}

int **mul(int **a, int **b, int size) {
  int **ret = malloc(size * sizeof(int *));
  for (int i = 0; i < size; i++) {
    ret[i] = calloc(size, sizeof(int));
    for (int j = 0; j < size; j++)
      for (int k = 0; k < size; k++)
        ret[i][j] += a[i][k] * b[k][j];
  }

  return ret;
}

int **array_mul(int ***data, int n, int size) {
  int **ret = data[0];
  #pragma omp parallel
  {
    #pragma omp single
    {
      for (int i = 1; i < n; i++) {
        #pragma omp task
        {
          int **result = mul(ret, data[i], size);
          free_matrix(data[i], size);
          ret = result;
        }
      }
    }
  }
  return ret;
}

int **rnd_matrix(int size) {
  int **ret = malloc(size * sizeof(int *));
  for (int i = 0; i < size; i++) {
    ret[i] = malloc(size * sizeof(int));
    for (int j = 0; j < size; j++)
      ret[i][j] = 2 * (rand() % 2) - 1; // Generates -1 or 1
  }

  return ret;
}

void print_matrix(int **m, int size) {
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++)
      printf("%d ", m[i][j]);
    printf("\n");
  }
}

int main(int argc, char **argv) {
  int n, size;
  double t;
  FILE *input;

  if (argc < 2) {
    fprintf(stderr, "Error: missing path to input file!\n");
    return EXIT_FAILURE;
  }

  if ((input = fopen(argv[1], "r")) == NULL) {
    fprintf(stderr, "Error: could not open input file!\n");
    return EXIT_FAILURE;
  }

  fscanf(input, "%d %d", &n, &size);
  srand(SEED);

  int ***data = malloc(n * sizeof(int **));
  for (int i = 0; i < n; i++)
    data[i] = rnd_matrix(size);

  t = omp_get_wtime();
  int **ret = array_mul(data, n, size);
  t = omp_get_wtime() - t;

  print_matrix(ret, size);
  fprintf(stderr, "%lf\n", t);

  free_matrix(ret, size); // Libera a ret
  free(data);
  return 0;
}
