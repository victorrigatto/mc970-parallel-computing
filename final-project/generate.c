/*

MC970 - Introdução à Programação Paralela - 2023.1

Projeto Final

Nome: Victor Rigatto
RA: 178068

Gerador de imagem aleatória input.txt para utilização como entrada

*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {

    int width, height;

    srand(time(NULL));

    fflush(stdin);

    printf("Insira a largura da imagem em pixels:\n");
    scanf("%d",&width);
    printf("Insira a altura da imagem em pixels:\n");
    scanf("%d",&height);

    FILE *file = fopen("input.txt", "w");
    if (file == NULL) {
        printf("Falha na geração do arquivo de entrada: input.txt\n");
        return 1;
    }

    fprintf(file, "%d %d\n", width, height);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int red = rand() % 256;
            int green = rand() % 256;
            int blue = rand() % 256;

            fprintf(file, "%d %d %d ", red, green, blue);
        }
        fprintf(file, "\n");
    }

    fclose(file);

    printf("Arquivo de entrada gerado: input.txt\n");

    return 0;
}
