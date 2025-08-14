/*
 * main.c - Driver do DGEMM (CAPBench style)
 *
 * CLI:
 *   --nthreads <int>
 *   --class {tiny,small,standard,large,huge}
 *   --block <int>           (0 = sem blocking; default configurável)
 *   --verbose
 *
 * Saída:
 *   "timing statistics" no padrão CAPBench + MFlops/s (média).
 */

#include <global.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <timer.h>
#include <util.h>
#include <math.h>
#include "dgemm.h"

/* Verbose e nthreads no padrão CAPBench */
int verbose = 0;
int nthreads = 1;

/* Problema: order e iterations por classe */
struct problem
{
    int order;
    int iterations;
};

/* Classes calibráveis (ajuste conforme sua máquina) */
static struct problem tiny = {256, 40};
static struct problem small = {512, 40};
static struct problem standard = {1024, 30};
static struct problem large = {2048, 20};
static struct problem huge = {3072, 12};

static struct problem *p = &tiny;

/* Options */
static int block_sz = 0; /* 0 = no blocking; adjust via --block */
static const double epsilon = 1.0e-8;

static void usage(void)
{
    printf("Usage: dgemm [options]\n");
    printf("Brief: Dense matrix-matrix multiplication (CAPBench style)\n");
    printf("Options:\n");
    printf("  --help                 Display this information and exit\n");
    printf("  --nthreads <value>     Set number of threads\n");
    printf("  --class <name>         tiny|small|standard|large|huge\n");
    printf("  --block <value>        Tile size (0 = no blocking)\n");
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
        else if (!strcmp(arg, "--block"))
        {
            if (i >= argc)
                usage();
            block_sz = atoi(argv[i++]);
            if (block_sz < 0)
                usage();
        }
        else
            usage();
    }
}

int main(int argc, char **argv)
{
    readargs(argc, argv);

    timer_init();
    srandnum(0);
    omp_set_num_threads(nthreads);

    if (verbose)
    {
        printf("initializing...\n");
        printf("  nthreads    : %d\n", nthreads);
        printf("  class       : order=%d, iterations=%d\n", p->order, p->iterations);
        printf("  block       : %d\n", block_sz);
    }

    /* DGEMM: executes iter=0..iterations (measuring 1..iterations) */
    double checksum = 0.0;
    double elapsed_sec = dgemm_kernel(p->order, p->iterations, block_sz, &checksum);

    /* PRK ref:
       ref_checksum = 0.25 * n^3 * (n-1)^2 * (iterations+1) */
    double n = (double)p->order;
    double ref_checksum = 0.25 * n * n * n * (n - 1.0) * (n - 1.0) * (double)(p->iterations + 1);

    /* Verification */
    if (fabs((checksum - ref_checksum) / ref_checksum) > epsilon)
    {
        printf("ERROR: Checksum = %lf, Reference checksum = %lf\n", checksum, ref_checksum);
        return 1;
    }
    else
    {
        printf("Solution validates\n");
#if VERBOSE
        printf("Reference checksum = %lf, checksum = %lf\n", ref_checksum, checksum);
#endif
    }

    /* Stats */
    double avg_time = elapsed_sec / (double)p->iterations;
    double nflops = 2.0 * n * n * n; /* por multiplicação C += A*B */
    printf("timing statistics:\n");
    printf("  total time:       %f\n", elapsed_sec * MICROSEC);
    printf("Rate (MFlops/s): %lf  Avg time (s): %lf\n",
           1.0e-06 * nflops / avg_time, avg_time);

    return 0;
}
