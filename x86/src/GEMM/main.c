// gemm_omp_blocked.c
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#ifndef ALIGN_BYTES
#define ALIGN_BYTES 64
#endif

#ifdef NOVEC
  #if defined(__clang__) || defined(__GNUC__)
    #define PRAGMA_NOVEC _Pragma("clang loop vectorize(disable)")
  #else
    #define PRAGMA_NOVEC
  #endif
#else
  #define PRAGMA_NOVEC
#endif

static inline void *xaligned_malloc(size_t n, size_t align) {
    void *p = NULL;
    if (posix_memalign(&p, align, n) != 0) return NULL;
    return p;
}

static inline double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int size_from_class(const char *cls) {
    if (!cls) return 1024;
    if (strcmp(cls, "tiny") == 0)     return 512;
    if (strcmp(cls, "standard") == 0) return 1024;
    if (strcmp(cls, "large") == 0)    return 2048;
    if (strcmp(cls, "huge") == 0)     return 4096;
    return 1024;
}

// Transpond B to BT (row-major) for stride-1 on k
static inline void transpose(int n, const double *restrict B, double *restrict BT) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            BT[(size_t)j*n + i] = B[(size_t)i*n + j];
}

// Microkernel 4x4: block compute C[ib:ib+4, jb:jb+4]
static inline void microkernel_4x4(
    int n, const double *restrict A, const double *restrict BT,
    double *restrict C, int ib, int jb, int kb, int BK)
{
    // 16 live acc → high register pressure
    double c00=0,c01=0,c02=0,c03=0;
    double c10=0,c11=0,c12=0,c13=0;
    double c20=0,c21=0,c22=0,c23=0;
    double c30=0,c31=0,c32=0,c33=0;

    // Ponteiros base para as 4 linhas de A e 4 linhas de BT (colunas de B)
    const double *a0 = &A[(size_t)ib*n + kb];
    const double *a1 = &A[(size_t)(ib+1)*n + kb];
    const double *a2 = &A[(size_t)(ib+2)*n + kb];
    const double *a3 = &A[(size_t)(ib+3)*n + kb];

    const double *b0 = &BT[(size_t)jb*n + kb];     // BT[jb, kb..kb+BK)
    const double *b1 = &BT[(size_t)(jb+1)*n + kb];
    const double *b2 = &BT[(size_t)(jb+2)*n + kb];
    const double *b3 = &BT[(size_t)(jb+3)*n + kb];

    PRAGMA_NOVEC
    for (int k = 0; k < BK; ++k) {
        double a0k = a0[k], a1k = a1[k], a2k = a2[k], a3k = a3[k];
        double b0k = b0[k], b1k = b1[k], b2k = b2[k], b3k = b3[k];

        // 16 FMA escalar (o vetorizar pode fundir isso em vetores/FMA)
        c00 += a0k * b0k; c01 += a0k * b1k; c02 += a0k * b2k; c03 += a0k * b3k;
        c10 += a1k * b0k; c11 += a1k * b1k; c12 += a1k * b2k; c13 += a1k * b3k;
        c20 += a2k * b0k; c21 += a2k * b1k; c22 += a2k * b2k; c23 += a2k * b3k;
        c30 += a3k * b0k; c31 += a3k * b1k; c32 += a3k * b2k; c33 += a3k * b3k;
    }

    // Acumula no C real
    C[(size_t)ib*n + jb]     += c00; C[(size_t)ib*n + (jb+1)] += c01;
    C[(size_t)ib*n + (jb+2)] += c02; C[(size_t)ib*n + (jb+3)] += c03;

    C[(size_t)(ib+1)*n + jb]     += c10; C[(size_t)(ib+1)*n + (jb+1)] += c11;
    C[(size_t)(ib+1)*n + (jb+2)] += c12; C[(size_t)(ib+1)*n + (jb+3)] += c13;

    C[(size_t)(ib+2)*n + jb]     += c20; C[(size_t)(ib+2)*n + (jb+1)] += c21;
    C[(size_t)(ib+2)*n + (jb+2)] += c22; C[(size_t)(ib+2)*n + (jb+3)] += c23;

    C[(size_t)(ib+3)*n + jb]     += c30; C[(size_t)(ib+3)*n + (jb+1)] += c31;
    C[(size_t)(ib+3)*n + (jb+2)] += c32; C[(size_t)(ib+3)*n + (jb+3)] += c33;
}

// GEMM bloqueado: C += A * B  (usando BT=transpose(B))
static inline void gemm_blocked(int n,
    const double *restrict A, const double *restrict BT, double *restrict C)
{
    // Blocos — ajuste livre: maiores ↑ compute/intensidade; menores ↑ overhead
    const int BM = 128, BN = 128, BK = 256;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int ib = 0; ib < n; ib += 4) {
        for (int jb = 0; jb < n; jb += 4) {

            // Zera bloco 4×4 local (acumulação está dentro do microkernel)
            // Loop sobre o painel k (em passos BK)
            for (int kb = 0; kb < n; kb += BK) {
                int curBK = (kb + BK <= n) ? BK : (n - kb);
                // “Varre” o painel de 4 linhas × curBK e 4 colunas × curBK
                // Faz microkernel em steps de curBK
                microkernel_4x4(n, A, BT, C, ib, jb, kb, curBK);
            }
        }
    }
}

static void usage(const char *p) {
    fprintf(stderr, "Usage: %s --class {tiny|standard|large|huge} --nthreads N\n", p);
}

int main(int argc, char **argv) {
    int N = 1024, nthreads = 1; const char *cls = "standard";
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--class") == 0 && i + 1 < argc) cls = argv[++i];
        else if (strcmp(argv[i], "--nthreads") == 0 && i + 1 < argc) nthreads = atoi(argv[++i]);
    }
    if (nthreads <= 0) { usage(argv[0]); return 1; }
    N = size_from_class(cls);
    omp_set_num_threads(nthreads);

    size_t bytes = (size_t)N * (size_t)N * sizeof(double);
    double *A  = (double*)xaligned_malloc(bytes, ALIGN_BYTES);
    double *B  = (double*)xaligned_malloc(bytes, ALIGN_BYTES);
    double *BT = (double*)xaligned_malloc(bytes, ALIGN_BYTES);
    double *C  = (double*)xaligned_malloc(bytes, ALIGN_BYTES);
    if (!A || !B || !BT || !C) { fprintf(stderr, "Allocation failed\n");
        free(A); free(B); free(BT); free(C); return 1; }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N*N; ++i) {
        A[i] = (double)((i % 101) - 50) * 0.001;
        B[i] = (double)((i %  97) - 48) * 0.001;
        C[i] = 0.0;
    }

    // Transpõe B para BT (para ter stride-1 no k nas duas matrizes)
    double tT0 = now_s();
    transpose(N, B, BT);
    double tT1 = now_s();

    // Warm-up
    gemm_blocked(N, A, BT, C);

    // Mede
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N*N; ++i) C[i] = 0.0;

    double t0 = now_s();
    gemm_blocked(N, A, BT, C);
    double t1 = now_s();

    double checksum = 0.0;
    #pragma omp parallel for reduction(+:checksum) schedule(static)
    for (int i = 0; i < N*N; ++i) checksum += C[i];

    printf("gemm-blocked 4x4 OpenMP\n");
    printf("class: %s\n", cls);
    printf("threads: %d\n", nthreads);
    printf("N: %d\n", N);
    printf("transpose_time(s): %.6f\n", tT1 - tT0);
    printf("time(s): %.6f\n", t1 - t0);
    printf("checksum: %.10e\n", checksum);

    free(A); free(B); free(BT); free(C);
    return 0;
}
