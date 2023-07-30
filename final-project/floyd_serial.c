/*

MC970 - Introdução à Programação Paralela - 2023.1

Projeto Final

Nome: Victor Rigatto
RA: 178068

Processamento de Imagem

Programa 1: Aplicação de pontilhamento com método Floyd-Steinberg

Versão Serial

*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void floydSteinbergDithering(int *pixels, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
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
    floydSteinbergDithering(pixels, width, height);
    t1 = omp_get_wtime() - t1;

    // Writing the dithered image to output.txt
    FILE *outputFile = fopen("output_floyd_serial.txt", "w");
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

    printf("Arquivo de saída: output_floyd_serial.txt\n");

    fprintf(stderr, "Tempo de execução do Floyd-Steinberg Serial: %lf\n", t1);

    free(pixels);

    return 0;
}
