/*
Saída do runtime.csv no Ada Lovelace para 2 execuções

[v178068@lovelace 02-Histogram]$ cat runtime.csv
# Input,Serial time,Parallel time,Speedup
1,0.087213,0.001973,44.2032
2,0.136295,0.007520,18.1243
3,0.466446,0.029147,16.0032
4,0.460129,0.031749,14.4927
5,2.475421,0.065782,37.6306
1,0.087172,0.001976,44.1153
2,0.136044,0.007515,18.1029
3,0.466024,0.029150,15.9871
4,0.459922,0.031753,14.4843
5,2.476848,0.065775,37.6563

*/

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define COMMENT "Histogram_GPU"
#define RGB_COMPONENT_COLOR 255

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

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

__global__ void histogram_kernel(PPMPixel *image_data, int tam, float *histogram) {

  int thread = threadIdx.x + blockIdx.x * blockDim.x;
  
  // Divisão dos pixels para os threads
  if (thread < tam) {
    image_data[thread].red = (image_data[thread].red * 4) / 256;
    image_data[thread].green = (image_data[thread].green * 4) / 256;
    image_data[thread].blue = (image_data[thread].blue * 4) / 256;
    
    int i = image_data[thread].red * 16 + image_data[thread].green * 4 + image_data[thread].blue;
    
    // Garantindo atomicidade
    atomicAdd(&histogram[i], 1.0);
  }
}

int main(int argc, char *argv[]) {

  // Variáveis para o resultado do histograma no device e o tempo do kernel
  float *d_histogram;
  float ms;
  cudaEvent_t start, stop;
  
  if (argc < 2) {
    fprintf(stderr, "Error: missing path to input file\n");
    return 1;
  }

  PPMImage *image = readPPM(argv[1]);

  // Aloca tamanho da imagem
  int tam = image->x * image->y;
  float *h = (float *)malloc(sizeof(float) * 64);

  // Initialize histogram
  for (int i = 0; i < 64; i++)
    h[i] = 0.0;

  // Aloca memória no device
  PPMPixel *d_image_data;
  cudaMalloc((void**)&d_image_data, sizeof(PPMPixel) * tam);
  cudaMalloc((void**)&d_histogram, sizeof(float) * 64);

  // Transfere a imagem inteira para o device
  cudaMemcpy(d_image_data, image->data, sizeof(PPMPixel) * tam, cudaMemcpyHostToDevice);
  
  int block_size = 256;
  int grid_size = (tam + block_size - 1) / block_size;

  // Função "Histogram" de chamada do kernel com marcação de tempo
  CUDACHECK(cudaEventCreate(&start));
  CUDACHECK(cudaEventCreate(&stop));
  CUDACHECK(cudaEventRecord(start));
  histogram_kernel<<<grid_size, block_size>>>(d_image_data, tam, d_histogram);
  CUDACHECK(cudaEventRecord(stop));
  CUDACHECK(cudaEventSynchronize(stop));
  CUDACHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDACHECK(cudaEventDestroy(start));
  CUDACHECK(cudaEventDestroy(stop));

  // Transfere o resultado para o host
  cudaMemcpy(h, d_histogram, sizeof(float) * 64, cudaMemcpyDeviceToHost);

  // Libera memória no device
  cudaFree(d_image_data);
  cudaFree(d_histogram);

  float n = image->x * image->y;

  for (int i = 0; i < 64; i++) {
    h[i] /= n;
  }

  for (int i = 0; i < 64; i++)
    printf("%0.3f ", h[i]);
  printf("\n");

  fprintf(stderr, "%lf\n", (((double)ms) / 1000.0));
  free(h);
}
