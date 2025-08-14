/*
 * main.c - Driver do SpMV (stencil 2D periódico) no padrão CAPBench
 *
 * CLI:
 *   --nthreads <int>
 *   --class {tiny,small,standard,large,huge}
 *   [--lsize <int>]          # sobrescreve a classe (size = 2^lsize)
 *   [--radius <int>]         # sobrescreve a classe
 *   [--scramble]             # ativa bit-reversal dos índices
 *   [--verbose]
 *
 * Saída:
 *   - "timing statistics" com total time (µs)
 *   - "Solution validates" (ou erro)
 *   - taxa "Rate (MFlops/s)" e tempo médio por iteração
 */

#include <global.h>
#include <omp.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <timer.h>
#include <util.h>
#include <math.h>
#include "sparse.h"

/* Verbose e nthreads no estilo CAPBench */
int verbose = 0;
int nthreads = 1;

/* Problema: lsize (log2 do lado), radius e iterações por classe */
struct problem
{
    int lsize;
    int radius;
    int iterations;
};

/* Classes calibráveis (valores seguros — ajuste se quiser mais carga) */
static struct problem tiny = {8, 1, 80};      /* size=256,  ordem=65.536 */
static struct problem small = {9, 1, 80};     /* size=512,  ordem=262.144 */
static struct problem standard = {10, 1, 50}; /* size=1024, ordem=1.048.576 */
static struct problem large = {11, 1, 30};    /* size=2048, ordem=4.194.304 */
static struct problem huge = {12, 1, 24};     /* size=4096, ordem=16.777.216 */

static struct problem *p = &tiny;

/* overrides opcionais */
static int lsize_override = -1;
static int radius_override = -1;
static int scramble_flag = 0;

static void usage(void)
{
    printf("Usage: sparse [options]\n");
    printf("Brief: Sparse matrix-vector (2D star stencil, periodic) — CAPBench style\n");
    printf("Options:\n");
    printf("  --help                 Display this information and exit\n");
    printf("  --nthreads <value>     Set number of threads\n");
    printf("  --class <name>         tiny|small|standard|large|huge\n");
    printf("  --lsize <value>        Override log2(grid side); size = 1<<lsize\n");
    printf("  --radius <value>       Override stencil radius (>=0)\n");
    printf("  --scramble             Enable bit-reversal scrambling of indices\n");
    printf("  --verbose              Be verbose\n");
    exit(0);
}

static void readargs(int argc, char **argv)
{
    int i = 1;
    while (i < argc)
    {
        const char *arg = argv[i++];
        if (!strcmp(arg, "--help"))
            usage();
        else if (!strcmp(arg, "--verbose"))
        {
            verbose = 1;
        }
        else if (!strcmp(arg, "--scramble"))
        {
            scramble_flag = 1;
        }
        else if (!strcmp(arg, "--nthreads"))
        {
            if (i >= argc)
                usage();
            nthreads = atoi(argv[i++]);
            if (nthreads < 1)
                usage();
        }
        else if (!strcmp(arg, "--class"))
        {
            if (i >= argc)
                usage();
            const char *c = argv[i++];
            if (!strcmp(c, "tiny"))
                p = &tiny;
            else if (!strcmp(c, "small"))
                p = &small;
            else if (!strcmp(c, "standard"))
                p = &standard;
            else if (!strcmp(c, "large"))
                p = &large;
            else if (!strcmp(c, "huge"))
                p = &huge;
            else
                usage();
        }
        else if (!strcmp(arg, "--lsize"))
        {
            if (i >= argc)
                usage();
            lsize_override = atoi(argv[i++]);
            if (lsize_override < 0)
                usage();
        }
        else if (!strcmp(arg, "--radius"))
        {
            if (i >= argc)
                usage();
            radius_override = atoi(argv[i++]);
            if (radius_override < 0)
                usage();
        }
        else
            usage();
    }
}

int main(int argc, char **argv)
{
    readargs(argc, argv);

    /* Define parâmetros efetivos */
    int lsize = (lsize_override >= 0) ? lsize_override : p->lsize;
    int radius = (radius_override >= 0) ? radius_override : p->radius;
    int iterations = p->iterations;

    /* Validações básicas (mesmas do PRK) */
    if (lsize < 0)
    {
        printf("ERROR: Log of grid size must be >= 0: %d\n", lsize);
        return 1;
    }
    const int size = 1 << lsize;
    if (size < 2 * radius + 1)
    {
        printf("ERROR: Grid extent %d smaller than stencil diameter %d\n",
               size, 2 * radius + 1);
        return 1;
    }

    timer_init();
    srandnum(0);
    omp_set_num_threads(nthreads);

    const int64_t size2 = (int64_t)size * (int64_t)size;
    const int stencil_size = 4 * radius + 1;
    const int64_t nent = size2 * (int64_t)stencil_size;
    const double sparsity = (double)stencil_size / (double)size2;

    if (verbose)
    {
        printf("initializing...\n");
        printf("  nthreads     : %d\n", nthreads);
        printf("  grid size    : size=%d (lsize=%d), order=%" PRId64 "\n", size, lsize, size2);
        printf("  stencil      : radius=%d, diameter=%d, nnz/row=%d\n",
               radius, 2 * radius + 1, stencil_size);
        printf("  sparsity     : %.10f\n", sparsity);
        printf("  iterations   : %d (timed)\n", iterations);
        printf("  scramble     : %s\n", scramble_flag ? "on" : "off");
    }

    /* Executa kernel */
    double vector_sum = 0.0;
    int64_t nent_kernel = 0;
    double elapsed_sec = sparse_kernel(lsize, radius, iterations, scramble_flag,
                                       &vector_sum, &nent_kernel);

    /* Checagem de consistência (opcional) */
    if (nent_kernel != nent)
    {
        printf("WARNING: nent mismatch: %" PRId64 " vs %" PRId64 "\n",
               nent_kernel, nent);
    }

    /* Verificação do PRK:
       reference_sum = 0.5 * nent * (iterations+1)*(iterations+2) */
    const double ref = 0.5 * (double)nent * (double)(iterations + 1) * (double)(iterations + 2);
    const double epsilon = 1.0e-8;

    if (fabs(vector_sum - ref) > epsilon)
    {
        printf("ERROR: Vector sum = %lf, Reference vector sum = %lf\n", vector_sum, ref);
        return 1;
    }
    else
    {
        printf("Solution validates\n");
#if VERBOSE
        printf("Reference sum = %lf, vector sum = %lf\n", ref, vector_sum);
#endif
    }

    /* Saídas no padrão CAPBench + taxa (PRK) */
    double avg_time = elapsed_sec / (double)iterations;
    /* custo por iteração: 2*nent FLOPs (1 mul + 1 add por nnz) */
    double mflops = 1.0e-06 * (2.0 * (double)nent) / avg_time;

    printf("timing statistics:\n");
    printf("  total time:       %f\n", elapsed_sec * MICROSEC);

    printf("Rate (MFlops/s): %lf  Avg time (s): %lf\n", mflops, avg_time);

    return 0;
}
