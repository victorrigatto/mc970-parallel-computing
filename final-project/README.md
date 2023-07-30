# MC970 - Introdução à Programação Paralela - 2023.1

## Projeto Final
## Explorando o Paralelismo no Processamento de Imagens

Nome: Victor Rigatto
RA: 178068

==============================================

## Descrição dos Arquivos

### Aplicação de blur com método convolucional gaussiano 3x3

Arquivo fonte serial -> blur_serial.c

Arquivo fonte paralelo -> blur_parallel.c (paralelização de loops de multiplicação de matrizes)

### Aplicação de pontilhamento com método Floyd-Steinberg

Arquivo fonte serial -> floyd_serial.c

Arquivo fonte paralelo abordagem 1 -> floyd_parallel_1.c (paralelização limitada por dependências)

Arquivo fonte paralelo abordagem 2 -> floyd_parallel_2.c (divisão dos pixels em blocos de trapézios)

### Geração de um arquivo de entrada aleatório

Arquivo fonte -> generate.c

==============================================

## Como Executar

Compilador utilizado: GCC 11.3.0 OpenMP 4.5
Perfilador utilizado: perf 5.15.90.1

Faça o download do repositório.
Todos os arquivos devem permanecer no mesmo diretório.
Execute a sequência a seguir para compilar todos os programas em sequência.

```sh
gcc generate.c -o generate
gcc blur_serial.c -o blur_serial -fopenmp
gcc blur_parallel.c -o blur_parallel -fopenmp
gcc floyd_serial.c -o floyd_serial -fopenmp
gcc floyd_parallel_1.c -o floyd_parallel_1 -fopenmp
gcc floyd_parallel_2.c -o floyd_parallel_2 -fopenmp -lm
```

Agora vamos criar nosso arquivo de entrada.

```sh
./generate
```
O gerador perguntará o tamanho da imagem em pixels desejado. Um valor de 4096 x 4096 parece bom. O arquivo input.txt será criado no diretório.
Agora podemos executar cada um dos programas separadamente, utilizando a mesma entrada, por razões de maior controle de perfilamento e análise.

```sh
./blur_serial
./blur_parallel
./floyd_serial
./floyd_parallel_1
./floyd_parallel_2
```

Cada programa exibirá, no final, o seu tempo de execução e o arquivo de saída que criou no diretório.

Podemos também executar os programas atrelados ao nperf, com repetições e visualizando o que ocorre na hierarquia de memória.

```sh
perf stat --repeat 2 -e cycles:u,instructions:u,cache-misses:u,L1-dcache-loads:u,L1-dcache-load-misses:u,L1-dcache-stores,dTLB-loads,dTLB-load-misses ./blur_serial
perf stat --repeat 2 -e cycles:u,instructions:u,cache-misses:u,L1-dcache-loads:u,L1-dcache-load-misses:u,L1-dcache-stores,dTLB-loads,dTLB-load-misses ./blur_parallel
perf stat --repeat 2 -e cycles:u,instructions:u,cache-misses:u,L1-dcache-loads:u,L1-dcache-load-misses:u,L1-dcache-stores,dTLB-loads,dTLB-load-misses ./floyd_serial
perf stat --repeat 2 -e cycles:u,instructions:u,cache-misses:u,L1-dcache-loads:u,L1-dcache-load-misses:u,L1-dcache-stores,dTLB-loads,dTLB-load-misses ./floyd_parallel_1
perf stat --repeat 2 -e cycles:u,instructions:u,cache-misses:u,L1-dcache-loads:u,L1-dcache-load-misses:u,L1-dcache-stores,dTLB-loads,dTLB-load-misses ./floyd_parallel_2
```

Também é interessante monitorar a utilização dos núcleos e threads da CPU.
