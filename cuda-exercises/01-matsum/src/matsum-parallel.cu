/*
Saída do runtime.csv no Ada Lovelace para 2 execuções

[v178068@lovelace 01-MatSum]$ cat runtime.csv
# Input,Serial time,Parallel time,Speedup
1,0.000012,0.000028,.4285
2,0.000934,0.000070,13.3428
3,0.002378,0.000075,31.7066
4,0.023240,0.000434,53.5483
5,0.122845,0.002130,57.6737
1,0.000015,0.000033,.4545
2,0.000881,0.000073,12.0684
3,0.002312,0.000075,30.8266
4,0.023100,0.000438,52.7397
5,0.123237,0.002112,58.3508

*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Kernel para divisão das somas das matrizes nos threads
__global__ void matrix_sum(int *A, int *B, int *C, int linhas, int colunas) {
    // TODO: Implement this kernel!
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < linhas && j < colunas) {
        int tam = i * colunas + j;
        C[tam] = A[tam] + B[tam];
    }
}

int main(int argc, char **argv) {
    int *A_host, *B_host, *C_host; // Variáveis da matriz no host
    int *A_device, *B_device, *C_device; // Variáveis da matriz no device
    int i, j;
    double t;

    // Input
    int linhas, colunas;
    FILE *input;

    if (argc < 2) {
        fprintf(stderr, "Error: missing path to input file\n");
        return EXIT_FAILURE;
    }

    if ((input = fopen(argv[1], "r")) == NULL) {
        fprintf(stderr, "Error: could not open file\n");
        return EXIT_FAILURE;
    }

    fscanf(input, "%d", &linhas);
    fscanf(input, "%d", &colunas);

    // Aloca memória no host
    A_host = (int *)malloc(sizeof(int) * linhas * colunas);
    B_host = (int *)malloc(sizeof(int) * linhas * colunas);
    C_host = (int *)malloc(sizeof(int) * linhas * colunas);

    // Inicializa memória no host
    for (i = 0; i < linhas; i++) {
        for (j = 0; j < colunas; j++) {
            A_host[i * colunas + j] = B_host[i * colunas + j] = i + j;
        }
    }

    // Aloca memória no device
    cudaMalloc((void **)&A_device, size);
    cudaMalloc((void **)&B_device, size);
    cudaMalloc((void **)&C_device, size);

    // Transfere memória do host para o device
    cudaMemcpy(A_device, A_host, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B_host, size, cudaMemcpyHostToDevice);

    // Blocos para aumentar o paralelismo na GPU
    dim3 blockSize(16, 16);
    dim3 gridSize((colunas + blockSize.x - 1) / blockSize.x, (linhas + blockSize.y - 1) / blockSize.y);

    // Compute matrix sum on device
    // Leave only the kernel and synchronize inside the timing region!
    t = omp_get_wtime();
    matrixsum<<<gridSize, blockSize>>>(A_device, B_device, C_device, linhas, colunas);
    cudaDeviceSynchronize();
    t = omp_get_wtime() - t;

    // Transfere resultado de volta para o host
    cudaMemcpy(C_host, C_device, size, cudaMemcpyDeviceToHost);

    long long int sum = 0;

    // Keep this computation on the CPU
    for (i = 0; i < linhas; i++) {
        for (j = 0; j < colunas; j++) {
            sum += C_host[i * colunas + j];
        }
    }

    // Imprime o resultado e o tempo
    fprintf(stdout, "%lli\n", sum);
    fprintf(stderr, "%lf\n", t);

    // Libera memória no host e no device
    free(A_host);
    free(B_host);
    free(C_host);
    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);
}
