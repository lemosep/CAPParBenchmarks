#include <global.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <timer.h>
#include <util.h>
#include <math.h>
#include "dgemm.h"

/* A(i,j) = A[i + order*j]  (col-major) */
#define A_ELEM(A, order, i, j) ((A)[(i) + (order) * (j)])
#define B_ELEM(B, order, i, j) ((B)[(i) + (order) * (j)])
#define C_ELEM(C, order, i, j) ((C)[(i) + (order) * (j)])

static void init_AB_C(double *A, double *B, double *C, int order)
{
#pragma omp parallel for schedule(static)
    for (int j = 0; j < order; ++j)
        for (int i = 0; i < order; ++i)
        {
            A_ELEM(A, order, i, j) = (double)j;
            B_ELEM(B, order, i, j) = (double)j;
            C_ELEM(C, order, i, j) = 0.0;
        }
}

static double sumC(const double *C, int order)
{
    double s = 0.0;
#pragma omp parallel for reduction(+ : s) schedule(static)
    for (int j = 0; j < order; ++j)
        for (int i = 0; i < order; ++i)
            s += C_ELEM(C, order, i, j);
    return s;
}

double dgemm_kernel(int order, int iterations, int block, double *checksum_out)
{
    double *A = (double *)smalloc((size_t)order * order * sizeof(double));
    double *B = (double *)smalloc((size_t)order * order * sizeof(double));
    double *C = (double *)smalloc((size_t)order * order * sizeof(double));
    if (!A || !B || !C)
    {
        fprintf(stderr, "ERROR: failed to allocate A/B/C\n");
        if (A)
            free(A);
        if (B)
            free(B);
        if (C)
            free(C);
        if (checksum_out)
            *checksum_out = 0.0;
        return 0.0;
    }

    init_AB_C(A, B, C, order);

    uint64_t t0 = 0, t1 = 0;
    double elapsed_sec = 0.0;

    /* Kernel with/without blocking */
#pragma omp parallel
    {
        /* Temp Buffers per thread when in blocking */
        double *AA = NULL, *BB = NULL, *CC = NULL;

        if (block > 0)
        {
            size_t tile = (size_t)block * block;
            AA = (double *)smalloc(tile * sizeof(double));
            BB = (double *)smalloc(tile * sizeof(double));
            CC = (double *)smalloc(tile * sizeof(double));
            if (!AA || !BB || !CC)
            {
                fprintf(stderr, "ERROR: failed to allocate tiles on thread %d\n",
                        omp_get_thread_num());
#pragma omp cancel parallel
            }
        }

        for (int iter = 0; iter <= iterations; ++iter)
        {

            if (iter == 1)
            {
#pragma omp barrier
#pragma omp master
                t0 = timer_get();
            }

            if (block > 0)
            {
                /* Blocked: pack A(i,kk:kk+bk), B(kk:kk+bk,jj:jj+bj)^T, accumulates CC, returns in C */
#pragma omp for schedule(static)
                for (int jj = 0; jj < order; jj += block)
                {
                    int jmax = jj + block;
                    if (jmax > order)
                        jmax = order;
                    int j_extent = jmax - jj;

                    for (int kk = 0; kk < order; kk += block)
                    {
                        int kmax = kk + block;
                        if (kmax > order)
                            kmax = order;
                        int k_extent = kmax - kk;

                        /* BB(j,k) = B(kk+k, jj+j)  (transpõe o bloco de B) */
                        for (int j = 0, jg = jj; jg < jmax; ++jg, ++j)
                            for (int k = 0, kg = kk; kg < kmax; ++kg, ++k)
                                BB[j * block + k] = B_ELEM(B, order, kg, jg);

                        for (int ii = 0; ii < order; ii += block)
                        {
                            int imax = ii + block;
                            if (imax > order)
                                imax = order;
                            int i_extent = imax - ii;

                            /* AA(i,k) = A(ii+i, kk+k) */
                            for (int k = 0, kg = kk; kg < kmax; ++kg, ++k)
                                for (int i = 0, ig = ii; ig < imax; ++ig, ++i)
                                    AA[i * block + k] = A_ELEM(A, order, ig, kg);

                            /* Zera CC(i,j) */
                            for (int j = 0; j < j_extent; ++j)
                                for (int i = 0; i < i_extent; ++i)
                                    CC[i * block + j] = 0.0;

                            /* CC += AA * BB  (dim: i_extent x j_extent) */
                            for (int k = 0; k < k_extent; ++k)
                                for (int j = 0; j < j_extent; ++j)
                                {
                                    double bkj = BB[j * block + k];
                                    for (int i = 0; i < i_extent; ++i)
                                        CC[i * block + j] += AA[i * block + k] * bkj;
                                }

                            /* C(ii+i, jj+j) += CC(i,j) */
                            for (int j = 0, jg = jj; j < j_extent; ++j, ++jg)
                                for (int i = 0, ig = ii; i < i_extent; ++i, ++ig)
                                    C_ELEM(C, order, ig, jg) += CC[i * block + j];
                        }
                    }
                }
            }
            else
            {
                /* Non-blocking */
#pragma omp for collapse(2) schedule(static)
                for (int jg = 0; jg < order; ++jg)
                    for (int kg = 0; kg < order; ++kg)
                    {
                        double bkj = B_ELEM(B, order, kg, jg);
                        for (int ig = 0; ig < order; ++ig)
                            C_ELEM(C, order, ig, jg) += A_ELEM(A, order, ig, kg) * bkj;
                    }
            }
        } /* iter loop */

#pragma omp barrier
#pragma omp master
        {
            t1 = timer_get();
            elapsed_sec = timer_diff(t0, t1);
        }

        if (block > 0)
        {
            free(AA);
            free(BB);
            free(CC);
        }
    } /* parallel */

    if (checksum_out)
        *checksum_out = sumC(C, order);

    free(A);
    free(B);
    free(C);
    return elapsed_sec;
}
