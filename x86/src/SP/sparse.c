#include <global.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <timer.h>
#include <util.h>
#include <math.h>
#include "sparse.h"

/* Conveniências */
#define BITS_IN_BYTE 8

/* grade: size = 1<<lsize, ordem da matriz = size^2 */
static inline int64_t lin_index(int i, int j, int lsize)
{
    /* LIN(i,j) = i + (j<<lsize) */
    return (int64_t)i + ((int64_t)j << lsize);
}

/* bit-reversal com “shift” para manter no intervalo [0, 2^{lsize2}) */
static inline uint64_t reverse_bits(uint64_t x, int shift_in_bits)
{
    x = ((x >> 1) & 0x5555555555555555ULL) | ((x << 1) & 0xAAAAAAAAAAAAAAAAULL);
    x = ((x >> 2) & 0x3333333333333333ULL) | ((x << 2) & 0xCCCCCCCCCCCCCCCCULL);
    x = ((x >> 4) & 0x0F0F0F0F0F0F0F0FULL) | ((x << 4) & 0xF0F0F0F0F0F0F0F0ULL);
    x = ((x >> 8) & 0x00FF00FF00FF00FFULL) | ((x << 8) & 0xFF00FF00FF00FF00ULL);
    x = ((x >> 16) & 0x0000FFFF0000FFFFULL) | ((x << 16) & 0xFFFF0000FFFF0000ULL);
    x = ((x >> 32) & 0x00000000FFFFFFFFULL) | ((x << 32) & 0xFFFFFFFF00000000ULL);
    return (x >> (int)(sizeof(uint64_t) * BITS_IN_BYTE - shift_in_bits));
}

static inline int64_t maybe_reverse_u64(int64_t a, int shift_in_bits, int scramble)
{
    return scramble ? (int64_t)reverse_bits((uint64_t)a, shift_in_bits) : a;
}

/* comparador crescente para qsort de int64_t */
static int cmp_i64(const void *pa, const void *pb)
{
    int64_t a = *(const int64_t *)pa;
    int64_t b = *(const int64_t *)pb;
    return (a < b) ? -1 : (a > b);
}

double sparse_kernel(int lsize, int radius, int iterations, int scramble,
                     double *vector_sum_out, int64_t *nent_out)
{
    const int lsize2 = 2 * lsize;
    const int size = 1 << lsize;
    const int64_t size2 = (int64_t)size * (int64_t)size;
    const int stencil_size = 4 * radius + 1;
    const int64_t nent = size2 * (int64_t)stencil_size;

    if (nent_out)
        *nent_out = nent;

    /* Alocações:
       - matrix[nent] (valores)
       - colIndex[nent] (índices de coluna)
       - vector[size2], result[size2] (num único bloco) */
    double *matrix = (double *)smalloc((size_t)nent * sizeof(double));
    int64_t *colIndex = (int64_t *)smalloc((size_t)nent * sizeof(int64_t));
    double *vec_res = (double *)smalloc((size_t)(2 * size2) * sizeof(double));
    if (!matrix || !colIndex || !vec_res)
    {
        fprintf(stderr, "ERROR: allocation failed in sparse_kernel\n");
        if (matrix)
            free(matrix);
        if (colIndex)
            free(colIndex);
        if (vec_res)
            free(vec_res);
        if (vector_sum_out)
            *vector_sum_out = 0.0;
        return 0.0;
    }
    double *vector = vec_res;
    double *result = vec_res + size2;

    /* Inicializa vector e result */
#pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < size2; ++row)
        vector[row] = result[row] = 0.0;

    /* Constrói estrutura CSR “achatada” por linha (com ordenação por linha) */
#pragma omp parallel
    {
        /* Preenche colIndex por linha; depois ordena o segmento e atribui matrix */
#pragma omp for schedule(static)
        for (int64_t row = 0; row < size2; ++row)
        {
            int j = (int)(row / size);
            int i = (int)(row % size);
            int64_t base = row * (int64_t)stencil_size;

            /* centro */
            colIndex[base] = maybe_reverse_u64(lin_index(i, j, lsize), lsize2, scramble);

            /* vizinhos (estrela) em +r/-r nas direções x e y, módulo périodico */
            int64_t elm = base;
            for (int r = 1; r <= radius; ++r, elm += 4)
            {
                int ip = (i + r) % size;
                int im = (i - r + size) % size;
                int jp = (j + r) % size;
                int jm = (j - r + size) % size;

                colIndex[elm + 1] = maybe_reverse_u64(lin_index(ip, j, lsize), lsize2, scramble);
                colIndex[elm + 2] = maybe_reverse_u64(lin_index(im, j, lsize), lsize2, scramble);
                colIndex[elm + 3] = maybe_reverse_u64(lin_index(i, jp, lsize), lsize2, scramble);
                colIndex[elm + 4] = maybe_reverse_u64(lin_index(i, jm, lsize), lsize2, scramble);
            }

            /* ordena colunas dessa linha (acesso crescente) */
            qsort(&colIndex[base], (size_t)stencil_size, sizeof(int64_t), cmp_i64);

            /* valores: 1.0/(col+1) como no PRK */
            for (int t = 0; t < stencil_size; ++t)
            {
                int64_t c = colIndex[base + t];
                matrix[base + t] = 1.0 / (double)(c + 1);
            }
        }
    } /* parallel (construção) */

    /* Loop de iterações (iter=0 é warmup; mede iter=1..iterations) */
    uint64_t t0 = 0, t1 = 0;
    double elapsed_sec = 0.0;

#pragma omp parallel
    {
        double tmp; /* acumulador local por linha */

        for (int iter = 0; iter <= iterations; ++iter)
        {
            if (iter == 1)
            {
#pragma omp barrier
#pragma omp master
                t0 = timer_get();
            }

            /* atualiza vetor: vector[row] += (row+1) */
#pragma omp for schedule(static)
            for (int64_t row = 0; row < size2; ++row)
                vector[row] += (double)(row + 1);

            /* SpMV: result[row] += sum_k (A[row,k] * vector[k]) */
#pragma omp for schedule(static)
            for (int64_t row = 0; row < size2; ++row)
            {
                int64_t first = row * (int64_t)stencil_size;
                int64_t last = first + (int64_t)stencil_size - 1;
                tmp = 0.0;
                /* pode-se experimentar #pragma omp simd reduction(+:tmp) */
                for (int64_t col = first; col <= last; ++col)
                    tmp += matrix[col] * vector[colIndex[col]];
                result[row] += tmp;
            }
        } /* iter loop */

#pragma omp barrier
#pragma omp master
        {
            t1 = timer_get();
            elapsed_sec = timer_diff(t0, t1);
        }
    } /* parallel (iterações) */

    /* soma final do vetor result (para verificação) */
    double vsum = 0.0;
#pragma omp parallel for reduction(+ : vsum) schedule(static)
    for (int64_t row = 0; row < size2; ++row)
        vsum += result[row];

    if (vector_sum_out)
        *vector_sum_out = vsum;

    free(matrix);
    free(colIndex);
    free(vec_res);

    return elapsed_sec;
}
