/*
 * main.c - Branch benchmark driver (CAPBench standard).
 *
 * CLI: --nthreads <int> --class {tiny,small,standard,large,huge}
 *      --branch {vector_stop,vector_go,no_vector}
 *      [--mode {with,without,both}] [--verbose]
 *
 * Output: "timing statistics" and checksum/validation.
 */

#include <global.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "timer.h"
#include <util.h>
#include "branch.h"

/* Verbose and nthreads in CAPBench style */
int verbose = 0;
int nthreads = 1;

/* Problem set (iterations, vector_length) */
struct problem
{
    int iterations;
    int vlen;
};

/* Selected to maintain sufficient work without becoming memory bound
 * given per-thread allocation; adjust according to your machine/study. */
static struct problem tiny = {20000, 8192};
static struct problem small = {40000, 16384};
static struct problem standard = {80000, 32768};
static struct problem large = {160000, 65536};
static struct problem huge = {320000, 131072};

static struct problem *p = &tiny;

/* Execution mode */
typedef enum
{
    MODE_WITH,
    MODE_WITHOUT,
    MODE_BOTH
} run_mode_t;
static run_mode_t mode = MODE_BOTH;

/* String to branch_type_t conversion */
static int parse_branch(const char *s, branch_type_t *out)
{
    if (!strcmp(s, "vector_stop"))
    {
        *out = BR_VECTOR_STOP;
        return 1;
    }
    if (!strcmp(s, "vector_go"))
    {
        *out = BR_VECTOR_GO;
        return 1;
    }
    if (!strcmp(s, "no_vector"))
    {
        *out = BR_NO_VECTOR;
        return 1;
    }
    if (!strcmp(s, "ins_heavy"))
    {
        *out = BR_INS_HEAVY;
        return 1;
    }
    return 0;
}

/* Usage */
static void usage(void)
{
    printf("Usage: branch [options]\n");
    printf("Brief: Branching benchmark (CAPBench style)\n");
    printf("Options:\n");
    printf("  --help                 Display this information and exit\n");
    printf("  --nthreads <value>     Set number of threads\n");
    printf("  --class <name>         Set problem class: tiny|small|standard|large|huge\n");
    printf("  --branch <name>        Branch type: vector_stop|vector_go|no_vector|ins_heavy\n");
    printf("  --mode <name>          Run: with|without|both (default: both)\n");
    printf("  --verbose              Be verbose\n");
    exit(0);
}

/* Read arguments in CAPBench style */
static void readargs(int argc, char **argv, branch_type_t *btype)
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
        else if (!strcmp(arg, "--branch"))
        {
            if (i >= argc)
                usage();
            if (!parse_branch(argv[i++], btype))
                usage();
        }
        else if (!strcmp(arg, "--mode"))
        {
            if (i >= argc)
                usage();
            const char *m = argv[i++];
            if (!strcmp(m, "with"))
                mode = MODE_WITH;
            else if (!strcmp(m, "without"))
                mode = MODE_WITHOUT;
            else if (!strcmp(m, "both"))
                mode = MODE_BOTH;
            else
                usage();
        }
        else
        {
            usage();
        }
    }
}

int main(int argc, char **argv)
{
    branch_type_t btype = BR_VECTOR_GO; /* reasonable default */

    readargs(argc, argv, &btype);

    timer_init();
    srandnum(0);
    omp_set_num_threads(nthreads);

    if (verbose)
    {
        printf("initializing...\n");
        printf("  nthreads    : %d\n", nthreads);
        printf("  class       : vlen=%d, iters=%d\n", p->vlen, p->iterations);
        printf("  branch      : %d\n", (int)btype);
        printf("  mode        : %s\n",
               mode == MODE_WITH ? "with" : (mode == MODE_WITHOUT ? "without" : "both"));
    }

    uint64_t t0, t1;
    double elapsed_with = 0.0, elapsed_without = 0.0;
    int nfunc = 0, rank = 0;
    int total_with = 0, total_without = 0;

    /* WITH BRANCHES */
    if (mode == MODE_WITH || mode == MODE_BOTH)
    {
        if (verbose)
            printf("running (with branches)...\n");
        t0 = timer_get();
        total_with = branch_with_branches(p->vlen, p->iterations, btype, &nfunc, &rank);
        t1 = timer_get();
        elapsed_with = timer_diff(t0, t1); /* em segundos */
    }

    /* WITHOUT BRANCHES */
    if (mode == MODE_WITHOUT || mode == MODE_BOTH)
    {
        if (verbose)
            printf("running (without branches)...\n");
        t0 = timer_get();
        total_without = branch_without_branches(p->vlen, p->iterations, btype, &nfunc, &rank);
        t1 = timer_get();
        elapsed_without = timer_diff(t0, t1);
    }

    /* Verification (same as PRK), based on effective number of threads */
    int nthreads_eff = nthreads; /* explicitly defined above */
    int vlen = p->vlen;
    int total_ref = ((vlen % 8) * ((vlen % 8) - 8) + vlen) / 2 * nthreads_eff;

    /* Outputs in CAPBench standard */
    if (mode == MODE_WITH || mode == MODE_BOTH)
    {
        printf("timing statistics (with branches):\n");
        printf("  total time:       %f\n", elapsed_with * MICROSEC);
        printf("  checksum:         %d (ref %d)\n", total_with, total_ref);
    }
    if (mode == MODE_WITHOUT || mode == MODE_BOTH)
    {
        printf("timing statistics (without branches):\n");
        printf("  total time:       %f\n", elapsed_without * MICROSEC);
        printf("  checksum:         %d (ref %d)\n", total_without, total_ref);
    }

    /* Validation message (useful in standardized scripts) */
    int ok_with = (mode != MODE_WITH) || (total_with == total_ref);
    int ok_without = (mode != MODE_WITHOUT) || (total_without == total_ref);
    if (ok_with && ok_without)
    {
        printf("Solution validates\n");
    }
    else
    {
        printf("ERROR: validation failed\n");
    }

    /* Extra information for INS_HEAVY (not supported) */
    if (btype == BR_INS_HEAVY)
    {
        printf("WARNING: ins_heavy not supported in this CAPBench adaptation.\n");
    }

    return 0;
}
