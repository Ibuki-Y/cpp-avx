#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <x86intrin.h>
// #include <immintrin.h>

void dgemm(int n, double *A, double *B, double *C) {
  for (int i = 0; i < n; i += 4) {
    for (int j = 0; j < n; ++j) {
      __m256d c0 = _mm256_load_pd(C + i + j * n); /* c0 = C[i][j] */
      for (int k = 0; k < n; ++k)
        c0 = _mm256_add_pd(c0, /* c0 += A[i][k]*B[k][j] */
                           _mm256_mul_pd(_mm256_load_pd(A + i + k * n),
                                         _mm256_broadcast_sd(B + k + j * n)));
      _mm256_store_pd(C + i + j * n, c0); /* C[i][j] = c0 */
    }
  }
}

int main(int argc, char *argv[]) {
  double *a, *b, *c;
  clock_t start, stop;
  int i, j, k, nn, n = 256; // 256, 512, 1024, 2048

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

  fprintf(stdout, "[AVX] elapsed time = %.5f [sec]\n",
          (double)(stop - start) / CLOCKS_PER_SEC);

  free(a);
  free(b);
  free(c);
  return 0;
}

/*
<AVX>
*/
