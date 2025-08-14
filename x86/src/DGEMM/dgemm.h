/*
 * dgemm.h - Dense matrix-matrix multiplication kernel
 */

#ifndef _DGEMM_H_
#define _DGEMM_H_

double dgemm_kernel(int order, int iterations, int block, double *checksum_out);

#endif