#include <stdio.h>
#include <stdlib.h>
#include <time.h>
// #include <x86intrin.h>
// #include <immintrin.h>
#define BLOCKSIZE 32

void do_block(int n, int si, int sj, int sk, double *A, double *B, double *C) {
  for (int i = si; i < si + BLOCKSIZE; ++i)
    for (int j = sj; j < sj + BLOCKSIZE; ++j) {
      double cij = C[i + j * n];
      for (int k = sk; k < sk + BLOCKSIZE; ++k)
        cij += A[i + k * n] * B[k + j * n];
      C[i + j * n] = cij;
    }
}

void dgemm(int n, double *A, double *B, double *C) {
  for (int sj = 0; sj < n; sj += BLOCKSIZE)
    for (int si = 0; si < n; si += BLOCKSIZE)
      for (int sk = 0; sk < n; sk += BLOCKSIZE)
        do_block(n, si, sj, sk, A, B, C);
}

int main(int argc, char *argv[]) {
  double *a, *b, *c;
  clock_t start, stop;
  int i, j, k, nn, n = 2048; // 256, 512, 1024, 2048

  if (argc > 1)
    n = atoi(argv[1]);
  fprintf(stdout, "matrix size = %d x %d\n", n, n);
  nn = n * n;

  a = (double *)malloc(sizeof(double) * nn);
  b = (double *)malloc(sizeof(double) * nn);
  c = (double *)malloc(sizeof(double) * nn);

  for (i = 0; i < nn; i++) {
    a[i] = (double)(rand() / 4096);
    b[i] = (double)(rand() / 4096);
    c[i] = 0;
  }

  start = clock();
  dgemm(n, a, b, c);
  stop = clock();

  fprintf(stdout, "[AVX+LU+CB] elapsed time = %.5f [sec]\n",
          (double)(stop - start) / CLOCKS_PER_SEC);

  free(a);
  free(b);
  free(c);
  return 0;
}

/*
<AVX+LU+CB>
matrix size = 256 x 256
[AVX+LU+CB] elapsed time = 0.09405 [sec]

matrix size = 512 x 512
[AVX+LU+CB] elapsed time = 0.48133 [sec]

matrix size = 1024 x 1024
[AVX+LU+CB] elapsed time = 3.65536 [sec]

matrix size = 2048 x 2048
[AVX+LU+CB] elapsed time = 29.28668 [sec]
*/
