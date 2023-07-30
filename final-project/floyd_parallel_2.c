/*

MC970 - Introdução à Programação Paralela - 2023.1

Projeto Final

Nome: Victor Rigatto
RA: 178068

Processamento de Imagem

Programa 1: Aplicação de pontilhamento com método Floyd-Steinberg

Versão Paralela 2

*/

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>

#define BLOCK_SIZE 16

typedef struct {
    int row;
    int col;
} PrimalBlock;

void getPrimalBlock(int M, int N, int a, int b, int itr, PrimalBlock *primalBlock) {
    if (itr > 0 && itr <= N / b) {
        primalBlock->row = 1;
        primalBlock->col = itr;
    } else {
        primalBlock->row = ceil((itr - (N / b)) / 2.0);
        primalBlock->col = (N / b) - ((itr - (N / b)) % 2);
    }
}

void ditherBlock(int row, int col, int *pixels, int width, int height) {
    int xStart = col * BLOCK_SIZE;
    int yStart = row * BLOCK_SIZE;
    int xEnd = xStart + BLOCK_SIZE;
    int yEnd = yStart + BLOCK_SIZE;

    for (int y = yStart; y < yEnd && y < height; y++) {
        for (int x = xStart; x < xEnd && x < width; x++) {
            if (x >= 0 && y >= 0) {
                int pixelIndex = y * width + x;
                int oldPixel = pixels[pixelIndex];
                int newPixel = oldPixel > 128 ? 255 : 0;
                pixels[pixelIndex] = newPixel;

                int quantError = oldPixel - newPixel;

                if (x + 1 < width)
                    pixels[pixelIndex + 1] += quantError * 7 / 16;
                if (x > 0 && y + 1 < height)
                    pixels[pixelIndex + width - 1] += quantError * 3 / 16;
                if (y + 1 < height)
                    pixels[pixelIndex + width] += quantError * 5 / 16;
                if (x + 1 < width && y + 1 < height)
                    pixels[pixelIndex + width + 1] += quantError * 1 / 16;
            }
        }
    }
}


int main() {
    double t1;
    FILE *file = fopen("input.txt", "r");
    if (file == NULL) {
        printf("Não foi possível abrir o arquivo de entrada.\n");
        printf("Utilize o generate para gerar um arquivo de entrada.\n");
        printf("O arquivo deve se chamar input.txt.\n");
        printf("O arquivo deve estar no mesmo diretório da aplicação.\n");
        return 1;
    }

    int width, height;
    fscanf(file, "%d %d", &width, &height);

    int totalPixels = width * height;
    int *pixels = malloc(totalPixels * sizeof(int));

    for (int i = 0; i < totalPixels; i++) {
        fscanf(file, "%d", &pixels[i]);
    }

    fclose(file);

    int M = height;
    int N = width;
    int a = BLOCK_SIZE;
    int b = BLOCK_SIZE;

    int numIterations = 2 * (M / a) + (N / b);

    t1 = omp_get_wtime();
    #pragma omp parallel
    {
    int numThreads = omp_get_num_threads();
    int threadId = omp_get_thread_num();
    int totalBlocks = (numIterations + numThreads - 1) / numThreads;  // Update calculation
    int start = threadId * totalBlocks;
    int end = (threadId + 1) * totalBlocks;
    if (end > numIterations) {
        end = numIterations;
    }

    for (int i = start; i < end; i++) {
        PrimalBlock primalBlock;
        getPrimalBlock(M, N, a, b, i, &primalBlock);

        for (int j = 0; j < totalBlocks; j++) {
            int row = primalBlock.row - j;
            int col = primalBlock.col - 2 * j;
            ditherBlock(row, col, pixels, width, height);
        }
    }
    }
    t1 = omp_get_wtime() - t1;


    // Writing the dithered image to output.txt
    FILE *outputFile = fopen("output_floyd_parallel_2.txt", "w");
    if (outputFile == NULL) {
        printf("Não foi possível salvar o arquivo de saída.");
        free(pixels);
        return 1;
    }

    fprintf(outputFile, "%d %d\n", width, height);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int pixelIndex = y * width + x;
            fprintf(outputFile, "%d ", pixels[pixelIndex]);
        }
        fprintf(outputFile, "\n");
    }

    fclose(outputFile);

    printf("Arquivo de saída: output_floyd_parallel_2.txt\n");

    fprintf(stderr, "Tempo de execução do Floyd-Steinberg Paralelo 2: %lf\n", t1);

    free(pixels);

    return 0;
}
