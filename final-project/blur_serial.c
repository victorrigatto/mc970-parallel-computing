/*

MC970 - Introdução à Programação Paralela - 2023.1

Projeto Final

Nome: Victor Rigatto
RA: 178068

Processamento de Imagem

Programa 2: Aplicação de blur com método convolucional gaussiano 3x3

Versão Serial

*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Function to perform convolution blur on the image
void convolutionBlur(int *pixels, int width, int height) {
    int kernel[3][3] = { {1, 2, 1},
                         {2, 4, 2},
                         {1, 2, 1} };
    int kernelSize = 3;
    int radius = kernelSize / 2;

    int *blurredPixels = malloc(width * height * sizeof(int));

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int sum = 0;
            for (int ky = -radius; ky <= radius; ky++) {
                for (int kx = -radius; kx <= radius; kx++) {
                    int nx = x + kx;
                    int ny = y + ky;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        int pixelIndex = ny * width + nx;
                        int kernelValue = kernel[ky + radius][kx + radius];
                        sum += pixels[pixelIndex] * kernelValue;
                    }
                }
            }
            int blurredPixelIndex = y * width + x;
            blurredPixels[blurredPixelIndex] = sum / 16; // Divide by the sum of kernel values (16 in this case)
        }
    }

    // Copy the blurred pixels back to the original array
    for (int i = 0; i < width * height; i++) {
        pixels[i] = blurredPixels[i];
    }

    free(blurredPixels);
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

    t1 = omp_get_wtime();
    convolutionBlur(pixels, width, height);
    t1 = omp_get_wtime() - t1;

    // Writing the dithered image to output.txt
    FILE *outputFile = fopen("output_blur_serial.txt", "w");
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

    printf("Arquivo de saída: output_blur_serial.txt\n");
    fprintf(stderr, "Tempo de execução do Blur: %lf\n", t1);

    free(pixels);

    return 0;
}
