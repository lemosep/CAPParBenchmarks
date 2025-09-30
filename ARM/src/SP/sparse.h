/*
 * sparse.h - Sparse matrix-vector (SpMV) kernel (CAPBench style)
 *
 * Baseado no PRK "sparse": matriz esparsa construída a partir de um
 * stencil estrela 2D periódico de raio 'radius' sobre uma grade de
 * lado 'size = 1<<lsize' (ordem = size^2), com opcional scrambling via
 * bit-reversal dos índices (quando 'scramble' != 0).
 */

#ifndef _SPARSE_H_
#define _SPARSE_H_

#include <stdint.h>

double sparse_kernel(int lsize, int radius, int iterations, int scramble,
                     double *vector_sum_out, int64_t *nent_out);

#endif /* _SPARSE_H_ */
