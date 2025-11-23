/*
 * main.c — Stencil driver (CAPBench style)
 *
 * CLI:
 *   --nthreads <int>
 *   --class {tiny,small,standard,large,huge}
 *   [--n <int>] [--it <int>] [--radius <int>]
 *   [--stencil star|compact]
 *   [--verbose]
 *
 * Output:
 *   - "Solution validates" (or error)
 *   - "timing statistics" (µs)
 *   - "Rate (MFlops/s)" and average time per iteration
 */

#include <global.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <timer.h>
#include <util.h>
#include "stencil.h"

/* Constants for double precision */
#define EPSILON 1.0e-8
#define COEFX 1.0
#define COEFY 1.0

int verbose = 0;
int nthreads = 1;

typedef struct
{
    int n, it, radius;
    stencil_type_t type;
} cls_t;
/* base classes (adjust per machine) */
static cls_t tiny = {1024, 200, 2, STENCIL_STAR};
static cls_t small = {2048, 150, 2, STENCIL_STAR};
static cls_t standard = {4096, 100, 2, STENCIL_STAR};
static cls_t large = {8192, 60, 3, STENCIL_COMPACT};
static cls_t huge = {16384, 40, 3, STENCIL_COMPACT};

static void usage(void)
{
    printf("Usage: stencil [options]\n");
    printf("Options:\n");
    printf("  --help\n");
    printf("  --nthreads <int>\n");
    printf("  --class {tiny,small,standard,large,huge}\n");
    printf("  --n <int>            # override grid size\n");
    printf("  --it <int>           # override iterations (timed)\n");
    printf("  --radius <int>\n");
    printf("  --stencil {star|compact}\n");
    printf("  --verbose\n");
    exit(0);
}

static stencil_type_t parse_type(const char *s)
{
    if (!strcmp(s, "star"))
        return STENCIL_STAR;
    if (!strcmp(s, "compact"))
        return STENCIL_COMPACT;
    usage();
    return STENCIL_STAR;
}

int main(int argc, char **argv)
{
    cls_t *C = &tiny;
    int n = 0, it = 0, radius = -1;
    stencil_type_t type = (stencil_type_t)-1;

    for (int i = 1; i < argc; ++i)
    {
        const char *a = argv[i];
        if (!strcmp(a, "--help"))
            usage();
        else if (!strcmp(a, "--verbose"))
            verbose = 1;
        else if (!strcmp(a, "--nthreads"))
        {
            if (++i >= argc)
                usage();
            nthreads = atoi(argv[i]);
        }
        else if (!strcmp(a, "--class"))
        {
            if (++i >= argc)
                usage();
            const char *c = argv[i];
            if (!strcmp(c, "tiny"))
                C = &tiny;
            else if (!strcmp(c, "small"))
                C = &small;
            else if (!strcmp(c, "standard"))
                C = &standard;
            else if (!strcmp(c, "large"))
                C = &large;
            else if (!strcmp(c, "huge"))
                C = &huge;
            else
                usage();
        }
        else if (!strcmp(a, "--n"))
        {
            if (++i >= argc)
                usage();
            n = atoi(argv[i]);
        }
        else if (!strcmp(a, "--it"))
        {
            if (++i >= argc)
                usage();
            it = atoi(argv[i]);
        }
        else if (!strcmp(a, "--radius"))
        {
            if (++i >= argc)
                usage();
            radius = atoi(argv[i]);
        }
        else if (!strcmp(a, "--stencil"))
        {
            if (++i >= argc)
                usage();
            type = parse_type(argv[i]);
        }
        else
        {
            usage();
        }
    }

    if (nthreads < 1)
        usage();

    /* apply class and overrides */
    if (n <= 0)
        n = C->n;
    if (it <= 0)
        it = C->it;
    if (radius < 0)
        radius = C->radius;
    if ((int)type < 0)
        type = C->type;

    /* validações fundamentais (como no PRK) */
    if (n < 1)
    {
        printf("ERROR: grid dimension must be positive: %d\n", n);
        return 1;
    }
    if (radius < 1)
    {
        printf("ERROR: radius must be >= 1: %d\n", radius);
        return 1;
    }
    if (2 * radius + 1 > n)
    {
        printf("ERROR: radius %d exceeds grid size %d\n", radius, n);
        return 1;
    }

    timer_init();
    omp_set_num_threads(nthreads);

    if (verbose)
    {
        printf("initializing...\n");
        printf("  nthreads          : %d\n", nthreads);
        printf("  grid size (n)     : %d\n", n);
        printf("  iterations (timed): %d\n", it);
        printf("  radius            : %d\n", radius);
        printf("  stencil type      : %s\n", (type == STENCIL_STAR ? "star" : "compact"));
        printf("  data type         : double\n");
    }

    double l1 = 0.0;
    double sec = stencil_kernel(n, radius, it, type, &l1);

    /* verificação (mesma do PRK): (iterations+1)*(COEFX+COEFY) */
    double ref = (double)(it + 1) * (COEFX + COEFY);
    if (fabs(l1 - ref) > EPSILON)
    {
        printf("ERROR: L1 norm = %lf, Reference L1 norm = %lf\n", l1, ref);
        return 1;
    }
    else
    {
#if VERBOSE
        printf("Reference L1 norm = %lf, L1 norm = %lf\n", ref, l1);
#endif
    }

    /* estatísticas no padrão CAPBench */
    double elapsed_with = sec;
    printf("  Runtime:       %f\n", elapsed_with * MICROSEC);

    return 0;
}
