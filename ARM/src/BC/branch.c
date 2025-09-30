/*
 * branch.c - Branching benchmark kernel (CAPBench style).
 *
 * Based on PRK "OpenMP Branching Bonanza" (Rob Van der Wijngaart, 2006),
 * restructured for CAPBench pattern: isolated kernel, local allocation,
 * internal OpenMP and checksum return.
 */

#include <global.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "branch.h"

/* Initialize vectors: alternating signs and identity index. */
static inline void init_vectors(int *vector, int *index, int n)
{
    for (int i = 0; i < n; ++i)
    {
        vector[i] = 3 - (i & 7);
        index[i] = i;
    }
}

/* Sum vector elements. */
static inline int sum_vector(const int *v, int n)
{
    int s = 0;
    for (int i = 0; i < n; ++i)
        s += v[i];
    return s;
}

int branch_with_branches(int vector_length, int iterations,
                         branch_type_t btype, int *nfunc, int *rank)
{
    if (iterations < 1 || (iterations % 2) != 0)
    {
        fprintf(stderr, "ERROR: iterations must be positive and even.\n");
        return 0;
    }
    if (btype == BR_INS_HEAVY)
    {
        fprintf(stderr, "ERROR: BR_INS_HEAVY not supported in this adaptation.\n");
        return 0;
    }

    int total = 0;

#pragma omp parallel reduction(+ : total)
    {
        int *vector = (int *)malloc(sizeof(int) * vector_length * 2);
        if (!vector)
        {
            fprintf(stderr, "ERROR: failed to allocate vector.\n");
#pragma omp cancel parallel
        }
        int *index = vector + vector_length;
        init_vectors(vector, index, vector_length);

        int i, iter, aux;

        switch (btype)
        {
        case BR_VECTOR_STOP:
            /* condition vector[index[i]]>0 inhibits vectorization */
            for (iter = 0; iter < iterations; iter += 2)
            {
#pragma omp for
                for (i = 0; i < vector_length; i++)
                {
                    aux = -(3 - (i & 7));
                    if (vector[index[i]] > 0)
                        vector[i] -= 2 * vector[i];
                    else
                        vector[i] -= 2 * aux;
                }
#pragma omp for
                for (i = 0; i < vector_length; i++)
                {
                    aux = (3 - (i & 7));
                    if (vector[index[i]] > 0)
                        vector[i] -= 2 * vector[i];
                    else
                        vector[i] -= 2 * aux;
                }
            }
            break;

        case BR_VECTOR_GO:
            /* condition aux>0 allows vectorization */
            for (iter = 0; iter < iterations; iter += 2)
            {
#pragma omp for
                for (i = 0; i < vector_length; i++)
                {
                    aux = -(3 - (i & 7));
                    if (aux > 0)
                        vector[i] -= 2 * vector[i];
                    else
                        vector[i] -= 2 * aux;
                }
#pragma omp for
                for (i = 0; i < vector_length; i++)
                {
                    aux = (3 - (i & 7));
                    if (aux > 0)
                        vector[i] -= 2 * vector[i];
                    else
                        vector[i] -= 2 * aux;
                }
            }
            break;

        case BR_NO_VECTOR:
            /* aux>0 would allow vectorization, but indirect indexing inhibits */
            for (iter = 0; iter < iterations; iter += 2)
            {
#pragma omp for
                for (i = 0; i < vector_length; i++)
                {
                    aux = -(3 - (i & 7));
                    if (aux > 0)
                        vector[i] -= 2 * vector[index[i]];
                    else
                        vector[i] -= 2 * aux;
                }
#pragma omp for
                for (i = 0; i < vector_length; i++)
                {
                    aux = (3 - (i & 7));
                    if (aux > 0)
                        vector[i] -= 2 * vector[index[i]];
                    else
                        vector[i] -= 2 * aux;
                }
            }
            break;

        default:
            break;
        }

        total += sum_vector(vector, vector_length);
        free(vector);
    }

    if (nfunc)
        *nfunc = 0;
    if (rank)
        *rank = 0;
    return total;
}

int branch_without_branches(int vector_length, int iterations,
                            branch_type_t btype, int *nfunc, int *rank)
{
    if (iterations < 1 || (iterations % 2) != 0)
    {
        fprintf(stderr, "ERROR: iterations must be positive and even.\n");
        return 0;
    }
    if (btype == BR_INS_HEAVY)
    {
        fprintf(stderr, "ERROR: BR_INS_HEAVY not supported in this adaptation.\n");
        return 0;
    }

    int total = 0;

#pragma omp parallel reduction(+ : total)
    {
        int *vector = (int *)malloc(sizeof(int) * vector_length * 2);
        if (!vector)
        {
            fprintf(stderr, "ERROR: failed to allocate vector.\n");
#pragma omp cancel parallel
        }
        int *index = vector + vector_length;
        init_vectors(vector, index, vector_length);

        int i, iter, aux;

        switch (btype)
        {
        case BR_VECTOR_STOP:
        case BR_VECTOR_GO:
            for (iter = 0; iter < iterations; iter += 2)
            {
#pragma omp for
                for (i = 0; i < vector_length; i++)
                {
                    aux = -(3 - (i & 7));
                    vector[i] -= (vector[i] + aux);
                }
#pragma omp for
                for (i = 0; i < vector_length; i++)
                {
                    aux = (3 - (i & 7));
                    vector[i] -= (vector[i] + aux);
                }
            }
            break;

        case BR_NO_VECTOR:
            for (iter = 0; iter < iterations; iter += 2)
            {
#pragma omp for
                for (i = 0; i < vector_length; i++)
                {
                    aux = -(3 - (i & 7));
                    vector[i] -= (vector[index[i]] + aux);
                }
#pragma omp for
                for (i = 0; i < vector_length; i++)
                {
                    aux = (3 - (i & 7));
                    vector[i] -= (vector[index[i]] + aux);
                }
            }
            break;

        default:
            break;
        }

        total += sum_vector(vector, vector_length);
        free(vector);
    }

    if (nfunc)
        *nfunc = 0;
    if (rank)
        *rank = 0;
    return total;
}
