#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <x86intrin.h>
// #include <immintrin.h>
#define UNROLL (4)
#define BLOCKSIZE 32

void do_block(int n, int si, int sj, int sk, double *A, double *B, double *C) {
  for (int i = si; i < si + BLOCKSIZE; i += UNROLL * 4) {
    for (int j = sj; j < sj + BLOCKSIZE; ++j) {
      __m256d c[4];
      for (int x = 0; x < UNROLL; ++x)
        c[x] = _mm256_load_pd(C + i + x * 4 + j * n);
      for (int k = sk; k < sk + BLOCKSIZE; ++k) {
        __m256d b = _mm256_broadcast_sd(B + k + j * n);
        for (int x = 0; x < UNROLL; ++x)
          c[x] = _mm256_add_pd(
              c[x], _mm256_mul_pd(_mm256_load_pd(A + n * k + x * 4 + i), b));
      }
      for (int x = 0; x < UNROLL; ++x)
        _mm256_store_pd(C + i + x * 4 + j * n, c[x]);
    }
  }
}

void dgemm_avx_lu_cb(int n, double *A, double *B, double *C) {
  int sj;
#pragma omp parallel for
  for (sj = 0; sj < n; sj += BLOCKSIZE)
    for (int si = 0; si < n; si += BLOCKSIZE)
      for (int sk = 0; sk < n; sk += BLOCKSIZE)
        do_block(n, si, sj, sk, A, B, C);
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
  dgemm_avx_lu_cb(n, a, b, c);
  stop = clock();

  fprintf(stdout, "[AVX+LU+CB+OMP] elapsed time = %.5f [sec]\n",
          (double)(stop - start) / CLOCKS_PER_SEC);

  free(a);
  free(b);
  free(c);
  return 0;
}

/*
<AVX+LU+CB+OMP>
*/
