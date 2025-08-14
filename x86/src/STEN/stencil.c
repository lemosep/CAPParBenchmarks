/*
 * stencil.c — CAPBench stencil kernel implementation
 *
 * Precision:
 *   - Uses double (like PRK when DOUBLE=1).
 *   - COEFX=COEFY=1.0; IN(i,j)=COEFX*i+COEFY*j in initialization.
 *
 * Timing:
 *   - iter=0 is warm-up; we measure iter=1..iterations with timer_get/diff.
 *
 * Verification:
 *   - Average L1 norm in interior should be (iterations+1)*(COEFX+COEFY).
 */

#include <global.h>
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <timer.h>
#include <util.h>
#include "stencil.h"

#define DTYPE double
#define COEFX 1.0
#define COEFY 1.0
#define EPSILON 1.0e-8                   /* indexers */
#define A_IDX(i, j, n) ((i) + (n) * (j)) /* build weights STAR: 4*R+1 points (discrete divergence) */
static void build_weights_star(DTYPE *weight, int R)
{
    int W = 2 * R + 1;
    /* zeros */
    for (int jj = -R; jj <= R; ++jj)
        for (int ii = -R; ii <= R; ++ii)
            weight[(ii + R) + W * (jj + R)] = 0.0;

    for (int ii = 1; ii <= R; ++ii)
    {
        DTYPE v = (DTYPE)(1.0 / (2.0 * ii * R));
        weight[(0 + R) + W * (ii + R)] = v;   /* (0,+ii) */
        weight[(ii + R) + W * (0 + R)] = v;   /* (+ii,0) */
        weight[(0 + R) + W * (-ii + R)] = -v; /* (0,-ii) */
        weight[(-ii + R) + W * (0 + R)] = -v; /* (-ii,0) */
    }
}

/* build weights COMPACT: (2R+1)^2 points (discrete rotational/grad) */
static void build_weights_compact(DTYPE *weight, int R)
{
    int W = 2 * R + 1;
    /* zeros */
    for (int jj = -R; jj <= R; ++jj)
        for (int ii = -R; ii <= R; ++ii)
            weight[(ii + R) + W * (jj + R)] = 0.0;

    for (int jj = 1; jj <= R; ++jj)
    {
        DTYPE base = (DTYPE)(1.0 / (4.0 * jj * (2.0 * jj - 1.0) * R));
        for (int ii = -jj + 1; ii < jj; ++ii)
        {
            weight[(ii + R) + W * (jj + R)] = base;
            weight[(ii + R) + W * (-jj + R)] = -base;
            weight[(jj + R) + W * (ii + R)] = base;
            weight[(-jj + R) + W * (ii + R)] = -base;
        }
        weight[(jj + R) + W * (jj + R)] = (DTYPE)(1.0 / (4.0 * jj * R));
        weight[(-jj + R) + W * (-jj + R)] = (DTYPE)(-1.0 / (4.0 * jj * R));
    }
}

double stencil_kernel(int n, int radius, int iterations,
                      stencil_type_t type,
                      double *l1_norm_out)
{
    if (n < 1)
    {
        fprintf(stderr, "ERROR: n must be >=1\n");
        exit(EXIT_FAILURE);
    }
    if (radius < 1)
    {
        fprintf(stderr, "ERROR: radius must be >=1\n");
        exit(EXIT_FAILURE);
    }
    if (2 * radius + 1 > n)
    {
        fprintf(stderr, "ERROR: radius exceeds grid size\n");
        exit(EXIT_FAILURE);
    }
    if (iterations < 1)
    {
        fprintf(stderr, "ERROR: iterations must be >=1\n");
        exit(EXIT_FAILURE);
    }

    size_t bytes = (size_t)n * (size_t)n * sizeof(DTYPE);
    DTYPE *in = (DTYPE *)smalloc(bytes);
    DTYPE *out = (DTYPE *)smalloc(bytes);
    if (!in || !out)
    {
        fprintf(stderr, "ERROR: alloc in/out\n");
        exit(EXIT_FAILURE);
    }

    /* weights (2R+1)*(2R+1) */
    int W = 2 * radius + 1;
    size_t wbytes = (size_t)W * (size_t)W * sizeof(DTYPE);
    DTYPE *weight = (DTYPE *)smalloc(wbytes);

    if (type == STENCIL_STAR)
        build_weights_star(weight, radius);
    else
        build_weights_compact(weight, radius);

/* initialize IN(i,j) = COEFX*i + COEFY*j; OUT=0 in interior */
#pragma omp parallel for schedule(static)
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            in[A_IDX(i, j, n)] = (DTYPE)(COEFX * (double)i + COEFY * (double)j);

#pragma omp parallel for schedule(static)
    for (int j = radius; j < n - radius; ++j)
        for (int i = radius; i < n - radius; ++i)
            out[A_IDX(i, j, n)] = 0.0;

    uint64_t t0 = 0, t1 = 0;
    double elapsed = 0.0;

    for (int iter = 0; iter <= iterations; ++iter)
    {

        if (iter == 1)
            t0 = timer_get();

        if (type == STENCIL_STAR)
        {
/* OUT += weights from 4*R+1 points */
#pragma omp parallel for schedule(static)
            for (int j = radius; j < n - radius; ++j)
            {
                for (int i = radius; i < n - radius; ++i)
                {
                    DTYPE acc = out[A_IDX(i, j, n)];
                    /* central column */
                    for (int jj = -radius; jj <= radius; ++jj)
                        acc += weight[(0 + radius) + W * (jj + radius)] * in[A_IDX(i, j + jj, n)];
                    /* central row (negatives and positives separated for readability) */
                    for (int ii = -radius; ii < 0; ++ii)
                        acc += weight[(ii + radius) + W * (0 + radius)] * in[A_IDX(i + ii, j, n)];
                    for (int ii = 1; ii <= radius; ++ii)
                        acc += weight[(ii + radius) + W * (0 + radius)] * in[A_IDX(i + ii, j, n)];
                    out[A_IDX(i, j, n)] = acc;
                }
            }
        }
        else
        {
/* OUT += weights from (2R+1)^2 */
#pragma omp parallel for schedule(static)
            for (int j = radius; j < n - radius; ++j)
            {
                for (int i = radius; i < n - radius; ++i)
                {
                    DTYPE acc = out[A_IDX(i, j, n)];
                    for (int jj = -radius; jj <= radius; ++jj)
                        for (int ii = -radius; ii <= radius; ++ii)
                            acc += weight[(ii + radius) + W * (jj + radius)] *
                                   in[A_IDX(i + ii, j + jj, n)];
                    out[A_IDX(i, j, n)] = acc;
                }
            }
        }

/* IN += 1.0 (entire grid) */
#pragma omp parallel for schedule(static)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                in[A_IDX(i, j, n)] += 1.0;
    }

    t1 = timer_get();
    elapsed = timer_diff(t0, t1);

    /* average L1 norm in interior */
    DTYPE norm = 0.0;
#pragma omp parallel for reduction(+ : norm) schedule(static)
    for (int j = radius; j < n - radius; ++j)
        for (int i = radius; i < n - radius; ++i)
            norm += fabs(out[A_IDX(i, j, n)]);

    DTYPE active = (DTYPE)(n - 2 * radius) * (DTYPE)(n - 2 * radius);
    norm /= active;
    if (l1_norm_out)
        *l1_norm_out = (double)norm;

    free(weight);
    free(out);
    free(in);

    return elapsed;
}
