#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void dgemm(int n, double *A, double *B, double *C) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      double cij = C[i + j * n]; /* cij = C[i][j] */
      for (int k = 0; k < n; k++)
        cij += A[i + k * n] * B[k + j * n]; /* cij += A[i][k]*B[k][j] */
      C[i + j * n] = cij;                   /* C[i][j] = cij */
    }
  }
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

  fprintf(stdout, "[Normal] elapsed time = %.5f [sec]\n",
          (double)(stop - start) / CLOCKS_PER_SEC);

  free(a);
  free(b);
  free(c);
  return 0;
}

/*
MacBook Air(Mid 2013)
プロセッサ 1.3GHz デュアルコアIntel Core i5
メモリ 4GB 1600MHz DDR3
*/

/*
<Normal>
matrix size = 256 x 256
[Normal] elapsed time = 0.16217 [sec]

matrix size = 512 x 512
[Normal] elapsed time = 2.34068 [sec]

matrix size = 1024 x 1024
[Normal] elapsed time = 32.41904 [sec]

matrix size = 2048 x 2048
[Normal] elapsed time = 245.65241 [sec]
*/
