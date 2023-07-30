/*
Perfilamento utilizando o Intel Core i3-8130u, 2 Cores 4 Threads

      N     Pi(N) 

  

         1         0 

         2         1 

         4         2 

         8         4 

        16         6 

        32        11 

        64        18 

       128        31 

       256        54 

       512        97 

      1024       172 

      2048       309 

      4096       564 

      8192      1028 

     16384      1900 

     32768      3512 

     65536      6542 

    131072     12251 

    262144     23000 

    524288     43390 

41.267835 

Performance counter stats for 'build/serial tests/5.in' (2 runs): 

      125392974629      cycles:u                                                      ( +-  2.11% )  (62.40%) 

      118202575063      instructions:u            #    0.96  insn per cycle           ( +-  0.04% )  (74.98%) 

            879027      cache-misses:u                                                ( +- 34.09% )  (75.02%) 

             65134      L1-dcache-loads                                               ( +-  1.20% )  (75.01%) 

            298144      L1-dcache-load-misses     #  463.30% of all L1-dcache accesses  ( +- 28.27% )  (74.95%) 

           4425144      L1-dcache-stores                                              ( +-  2.10% )  (75.04%) 

           7514093      dTLB-loads                                                    ( +-  2.10% )  (50.03%) 

            242590      dTLB-load-misses          #    3.30% of all dTLB cache accesses  ( +- 20.91% )  (49.95%) 

  

             38.65 +- 2.61 seconds time elapsed  ( +-  6.76% ) 


       N     Pi(N) 

  

         1         0 

         2         1 

         4         2 

         8         4 

        16         6 

        32        11 

        64        18 

       128        31 

       256        54 

       512        97 

      1024       172 

      2048       309 

      4096       564 

      8192      1028 

     16384      1900 

     32768      3512 

     65536      6542 

    131072     12251 

    262144     23000 

    524288     43390 

18.384380 

Performance counter stats for 'build/parallel tests/5.in' (2 runs): 

      152575406849      cycles:u                                                      ( +-  0.10% )  (62.38%) 

      118411851212      instructions:u            #    0.78  insn per cycle           ( +-  0.12% )  (74.94%) 

            263613      cache-misses:u                                                ( +- 34.33% )  (75.04%) 

           9153582      L1-dcache-loads                                               ( +-  1.06% )  (75.02%) 

            233586      L1-dcache-load-misses     #    2.52% of all L1-dcache accesses  ( +- 19.57% )  (75.05%) 

           4906048      L1-dcache-stores                                              ( +-  3.14% )  (75.12%) 

          14908383      dTLB-loads                                                    ( +-  0.30% )  (49.91%) 

            201726      dTLB-load-misses          #    1.36% of all dTLB cache accesses  ( +- 18.34% )  (49.88%) 

            18.834 +- 0.449 seconds time elapsed  ( +-  2.38% ) 

*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char *argv[]);
int prime_default(int n);

int main(int argc, char *argv[]) {
  int n;
  int n_factor;
  int n_hi;
  int n_lo;
  int primes;
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

  n_lo = 1;
  n_factor = 2;

  // Do not change this line
  omp_set_num_threads(4);

  fscanf(input, "%d", &n_hi);
  n_hi = 1 << n_hi;

  printf("                    \n");
  printf("         N     Pi(N)\n");
  printf("\n");

  n = n_lo;

  t = omp_get_wtime();

  while (n <= n_hi) {
    primes = prime_default(n);

    printf("  %8d  %8d\n", n, primes);

    n = n * n_factor;
  }

  t = omp_get_wtime() - t;

  /*
    Terminate.
  */
  fprintf(stderr, "%lf\n", t);

  return 0;
}

/*
  Purpose:
   counts primes.
  Licensing:
    This code is distributed under the GNU LGPL license.
  Modified:
    10 July 2010
  Author:
    John Burkardt
  Parameters:
    Input, the maximum number to check.
    Output, the number of prime numbers up to N.
*/
int prime_default(int n) {
  int i;
  int j;
  int prime;
  int total = 0;

  #pragma omp parallel for private(i, j, prime) reduction(+:total)
  for (i = 2; i <= n; i++) {
    prime = 1;

    for (j = 2; j < i; j++) {
      if (i % j == 0) {
        prime = 0;
        break;
      }
    }
    total = total + prime;
  }

  return total;
}
