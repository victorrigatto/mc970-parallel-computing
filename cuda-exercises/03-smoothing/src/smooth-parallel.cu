/*
Saída do runtime.csv no Ada Lovelace para 2 execuções

[v178068@lovelace 03-Smoothing]$ cat runtime.csv
# Input,Serial time,Parallel time,Speedup
1,0.365873,0.000547,668.8720
2,0.822896,0.001170,703.3299
3,3.290458,0.004428,743.1025
4,3.298119,0.004437,743.3218
5,11.462606,0.015032,762.5469
1,0.365435,0.000550,664.4272
2,0.824851,0.001177,700.8079
3,3.291954,0.004438,741.7652
4,3.297204,0.004430,744.2898
5,11.394218,0.015034,757.8966

*/

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define MASK_WIDTH 15

#define COMMENT "Histogram_GPU"
#define RGB_COMPONENT_COLOR 255

void check_cuda(cudaError_t error, const char *filename, const int line)
{
  if (error != cudaSuccess) {
    fprintf(stderr, "Error: %s:%d: %s: %s\n", filename, line,
                 cudaGetErrorName(error), cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}

#define CUDACHECK(cmd) check_cuda(cmd, __FILE__, __LINE__)

typedef struct {
  unsigned char red, green, blue;
} PPMPixel;

typedef struct {
  int x, y;
  PPMPixel *data;
} PPMImage;

static PPMImage *readPPM(const char *filename) {
  char buff[16];
  PPMImage *img;
  FILE *fp;
  int c, rgb_comp_color;
  fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, "Unable to open file '%s'\n", filename);
    exit(1);
  }

  if (!fgets(buff, sizeof(buff), fp)) {
    perror(filename);
    exit(1);
  }

  if (buff[0] != 'P' || buff[1] != '6') {
    fprintf(stderr, "Invalid image format (must be 'P6')\n");
    exit(1);
  }

  img = (PPMImage *)malloc(sizeof(PPMImage));
  if (!img) {
    fprintf(stderr, "Unable to allocate memory\n");
    exit(1);
  }

  c = getc(fp);
  while (c == '#') {
    while (getc(fp) != '\n')
      ;
    c = getc(fp);
  }

  ungetc(c, fp);
  if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
    fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
    exit(1);
  }

  if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
    fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
    exit(1);
  }

  if (rgb_comp_color != RGB_COMPONENT_COLOR) {
    fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
    exit(1);
  }

  while (fgetc(fp) != '\n')
    ;
  img->data = (PPMPixel *)malloc(img->x * img->y * sizeof(PPMPixel));

  if (!img) {
    fprintf(stderr, "Unable to allocate memory\n");
    exit(1);
  }

  if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
    fprintf(stderr, "Error loading image '%s'\n", filename);
    exit(1);
  }

  fclose(fp);
  return img;
}

void writePPM(PPMImage *img) {

  fprintf(stdout, "P6\n");
  fprintf(stdout, "# %s\n", COMMENT);
  fprintf(stdout, "%d %d\n", img->x, img->y);
  fprintf(stdout, "%d\n", RGB_COMPONENT_COLOR);

  fwrite(img->data, 3 * img->x, img->y, stdout);
  fclose(stdout);
}

// Kernel para dividir os loops de aplicação da máscara em blocos nos threads
__global__ void smoothing_kernel(const PPMPixel *input, PPMPixel *output, int width, int height) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < height && j < width) {
    int total_red = 0, total_blue = 0, total_green = 0;
    int count = 0;

    for (int y = i - ((MASK_WIDTH - 1) / 2); y <= (i + ((MASK_WIDTH - 1) / 2)); y++) {
      for (int x = j - ((MASK_WIDTH - 1) / 2); x <= (j + ((MASK_WIDTH - 1) / 2)); x++) {
        if (x >= 0 && y >= 0 && y < height && x < width) {
          total_red += input[(y * width) + x].red;
          total_blue += input[(y * width) + x].blue;
          total_green += input[(y * width) + x].green;
          count++;
        }
      }
    }

    output[(i * width) + j].red = total_red / (MASK_WIDTH * MASK_WIDTH);
    output[(i * width) + j].blue = total_blue / (MASK_WIDTH * MASK_WIDTH);
    output[(i * width) + j].green = total_green / (MASK_WIDTH * MASK_WIDTH);
  }
}

int main(int argc, char *argv[]) {
  FILE *input;
  char filename[255];
  double t;

  if (argc < 2) {
    fprintf(stderr, "Error: missing path to input file\n");
    return 1;
  }

  if ((input = fopen(argv[1], "r")) == NULL) {
    fprintf(stderr, "Error: could not open input file!\n");
    return 1;
  }

  // Chama para leitura da imagem
  fscanf(input, "%s\n", filename);
  PPMImage *image = readPPM(filename);
  PPMImage *image_output = readPPM(filename);

  // Aloca memória no device
  PPMPixel *d_input, *d_output;
  size_t imageSize = image->x * image->y * sizeof(PPMPixel);
  cudaMalloc((void **)&d_input, imageSize);
  cudaMalloc((void **)&d_output, imageSize);

  // Transfere memória do host para o device
  cudaMemcpy(d_input, image->data, imageSize, cudaMemcpyHostToDevice);

  // Tamanho dos blocos para melhorar eficiência do paralelismo na GPU e maior speedup
  dim3 blockSize(16, 16);
  dim3 gridSize((image->x + blockSize.x - 1) / blockSize.x, (image->y + blockSize.y - 1) / blockSize.y);

  // Marca o tempo e chama o kernel
  t = omp_get_wtime();
  smoothing_kernel<<<gridSize, blockSize>>>(d_input, d_output, image->x, image->y);
  cudaDeviceSynchronize();
  t = omp_get_wtime() - t;

  // Transfere resultado de volta para o host
  cudaMemcpy(image_output->data, d_output, imageSize, cudaMemcpyDeviceToHost);

  // Chama para imprimir o resultado
  writePPM(image_output);

  // Imprime o tempo
  fprintf(stderr, "%lf\n", t);

  // Libera memória no host e no device
  cudaFree(d_input);
  cudaFree(d_output);
  free(image->data);
  free(image);
  free(image_output->data);
  free(image_output);

  return 0;
}
