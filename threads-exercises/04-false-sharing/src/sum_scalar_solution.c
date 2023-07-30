/* 

Original:

./perf stat --repeat 2 -e cycles:u,instructions:u,cache-misses:u,L1-dcache-loads:u,L1-dcache-load-misses:u,L1-dcache-stores,dTLB-loads,dTLB-load-misses ./sum_scalar 3000000 4
sum = 6000000
time_us = 8605
sum = 6000000
time_us = 8051

 Performance counter stats for './sum_scalar 3000000 4' (2 runs):

          54246673      cycles:u                                                      ( +-  7.88% )
          45137317      instructions:u            #    0.77  insn per cycle           ( +-  0.00% )
              9613      cache-misses:u                                                ( +-  4.41% )
          18027754      L1-dcache-loads                                               ( +-  0.00% )
            299981      L1-dcache-load-misses     #    1.66% of all L1-dcache accesses  ( +-  8.65% )
           6119433      L1-dcache-stores                                              ( +-  0.00% )
     <not counted>      dTLB-loads                                                    (0.00%)
     <not counted>      dTLB-load-misses                                              (0.00%)

          0.009259 +- 0.000199 seconds time elapsed  ( +-  2.15% )


Otimizado:

./perf stat --repeat 2 -e cycles:u,instructions:u,cache-misses:u,L1-dcache-loads:u,L1-dcache-load-misses:u,L1-dcache-stores,dTLB-loads,dTLB-load-misses ./sum_scalar_solution 3000000 4
sum = 6000000
time_us = 3169
sum = 6000000
time_us = 3891

 Performance counter stats for './sum_scalar_solution 3000000 4' (2 runs):

          44743742      cycles:u                                                      ( +- 13.05% )
          72138284      instructions:u            #    1.85  insn per cycle           ( +-  0.00% )
              6002      cache-misses:u                                                ( +- 14.59% )
          24031858      L1-dcache-loads                                               ( +-  0.00% )
            120051      L1-dcache-load-misses     #    0.50% of all L1-dcache accesses  ( +- 44.18% )
           6120702      L1-dcache-stores                                              ( +-  0.01% )
     <not counted>      dTLB-loads                                                    (0.00%)
     <not counted>      dTLB-load-misses                                              (0.00%)

          0.004308 +- 0.000357 seconds time elapsed  ( +-  8.29% )

*/

/*
 * sum_scalar.c - A simple parallel sum program to sum a
 * series of scalars
 */

/*
 * sum_scalar.c - A simple parallel sum program to sum a
 * series of scalars
 */

#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define MAXTHREADS 8

#define CACHE_LINE_SIZE 64 // Assuming a cache line size of 64 bytes

void *sum(void *p);

// global shared variables
unsigned long int psum[MAXTHREADS][CACHE_LINE_SIZE/sizeof(unsigned long int)] __attribute__ ((aligned(CACHE_LINE_SIZE))); // partial sum computed by each thread
unsigned long int sumtotal = 0;
unsigned long int n;
int numthreads;
pthread_mutex_t mutex;

int main(int argc, char **argv) {
  pthread_t tid[MAXTHREADS];
  int i, myid[MAXTHREADS];
  struct timeval start, end;

  gettimeofday(&start, NULL); /* start timing */

  if (argc != 3) {
    printf("Usage: %s <n> <numthreads>\n", argv[0]);
    return 1;
  }

  n = strtoul(argv[1], NULL, 10);
  numthreads = (int)strtoul(argv[2], NULL, 10);

  for (i = 0; i < numthreads; i++) {
    myid[i] = i;
    for (int j = 0; j < CACHE_LINE_SIZE/sizeof(unsigned long int); j++) {
      psum[i][j] = 0.0;
    }
    pthread_create(&tid[i], NULL, sum, &myid[i]);
  }

  for (i = 0; i < numthreads; i++) {
    pthread_join(tid[i], NULL);
  }

  pthread_mutex_destroy(&mutex);
  gettimeofday(&end, NULL); /* end timing */
  long spent = (end.tv_sec * 1000000 + end.tv_usec) -
               (start.tv_sec * 1000000 + start.tv_usec);

  printf("sum = %lu\ntime_us = %ld\n", sumtotal, spent);

  return 0;
}

void *sum(void *p) {
  int myid = *((int *)p);
  unsigned long int start = (myid * (unsigned long int)n) / numthreads;
  unsigned long int end = ((myid + 1) * (unsigned long int)n) / numthreads;
  unsigned long int i;

  for (i = start; i < end; i++) {
    psum[myid][i%(CACHE_LINE_SIZE/sizeof(unsigned long int))] += 2;
  }

  unsigned long int partial_sum = 0;
  for (int j = 0; j < CACHE_LINE_SIZE/sizeof(unsigned long int); j++) {
    partial_sum += psum[myid][j];
  }

  pthread_mutex_lock(&mutex);
  sumtotal += partial_sum;
  pthread_mutex_unlock(&mutex);

  return NULL;
}
