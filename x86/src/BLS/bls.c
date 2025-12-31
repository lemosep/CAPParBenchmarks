// Copyright (c) 2007 Intel Corp.
// CAPBench-style adaptation (flags: --nthreads, --class)
// NOTE: synthetic input data, no file I/O.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <getopt.h>
#include <omp.h>

/* ---------------- Configuration ---------------- */

#define fptype float
#define NUM_RUNS 100

typedef enum {
  CLASS_TINY,
  CLASS_SMALL,
  CLASS_STANDARD,
  CLASS_LARGE,
  CLASS_HUGE
} bench_class_t;

typedef struct {
  int nthreads;
  bench_class_t cls;
  int blocksize;      // BSIZE
  int numOptions;
} options_t;

typedef struct OptionData_ {
  fptype s;          // spot price
  fptype strike;     // strike price
  fptype r;          // risk-free interest rate
  fptype divq;       // dividend rate (unused)
  fptype v;          // volatility
  fptype t;          // time to maturity
  char OptionType;   // 'P' or 'C'
  fptype divs;       // unused
  fptype DGrefval;   // unused unless ERR_CHK
} OptionData;

/* Globals (kept similar to original) */
static OptionData *data_g;
static int numOptions_g;

static int    *otype_g;
static fptype *sptprice_g;
static fptype *strike_g;
static fptype *rate_g;
static fptype *volatility_g;
static fptype *otime_g;

static int numError_g = 0;

/* ---------------- Utils ---------------- */

static void usage(const char *prog) {
  fprintf(stderr,
    "Usage: %s [--nthreads N] [--class tiny|small|standard|large|huge] [--blocksize B]\n"
    "Defaults: --nthreads=omp_get_max_threads(), --class=standard, --blocksize=1024\n",
    prog
  );
}

static int parse_class(const char *s, bench_class_t *out) {
  if (!s) return 0;
  if (strcmp(s, "tiny") == 0)     { *out = CLASS_TINY; return 1; }
  if (strcmp(s, "small") == 0)    { *out = CLASS_SMALL; return 1; }
  if (strcmp(s, "standard") == 0) { *out = CLASS_STANDARD; return 1; }
  if (strcmp(s, "large") == 0)    { *out = CLASS_LARGE; return 1; }
  if (strcmp(s, "huge") == 0)     { *out = CLASS_HUGE; return 1; }
  return 0;
}

/* Choose numOptions per class.
   You MUST tune these for your platform/time goals. */
static int class_numOptions(bench_class_t cls) {
  switch (cls) {
    case CLASS_TINY:     return 1 << 12; // 4096
    case CLASS_SMALL:    return 1 << 14; // 16384
    case CLASS_STANDARD: return 1 << 16; // 65536
    case CLASS_LARGE:    return 1 << 20; // 262144
    case CLASS_HUGE:     return 1 << 24; // 1048576
    default:             return 1 << 16;
  }
}

static options_t parse_args(int argc, char **argv) {
  options_t opt;
  opt.nthreads  = -1;
  opt.cls       = CLASS_STANDARD;
  opt.blocksize = 1024;

  static struct option long_opts[] = {
    {"nthreads",   required_argument, 0, 't'},
    {"class",      required_argument, 0, 'c'},
    {"blocksize",  required_argument, 0, 'b'},
    {"help",       no_argument,       0, 'h'},
    {0,0,0,0}
  };

  int ch;
  while ((ch = getopt_long(argc, argv, "t:c:b:h", long_opts, NULL)) != -1) {
    switch (ch) {
      case 't':
        opt.nthreads = atoi(optarg);
        if (opt.nthreads <= 0) {
          fprintf(stderr, "ERROR: --nthreads must be > 0\n");
          usage(argv[0]);
          exit(1);
        }
        break;
      case 'c':
        if (!parse_class(optarg, &opt.cls)) {
          fprintf(stderr, "ERROR: invalid --class '%s'\n", optarg);
          usage(argv[0]);
          exit(1);
        }
        break;
      case 'b':
        opt.blocksize = atoi(optarg);
        if (opt.blocksize <= 0) {
          fprintf(stderr, "ERROR: --blocksize must be > 0\n");
          exit(1);
        }
        break;
      case 'h':
      default:
        usage(argv[0]);
        exit(0);
    }
  }

  if (opt.nthreads < 0) opt.nthreads = omp_get_max_threads();
  opt.numOptions = class_numOptions(opt.cls);
  return opt;
}

/* Deterministic pseudo-rng */
static unsigned int lcg_u32(unsigned int *state) {
  *state = (*state * 1664525u) + 1013904223u;
  return *state;
}
static float frand01(unsigned int *state) {
  return (lcg_u32(state) >> 8) * (1.0f / 16777216.0f); // 24-bit mantissa-ish
}

/* ---------------- Black-Scholes kernel ---------------- */

#define inv_sqrt_2xPI 0.39894228040143270286f

static inline fptype CNDF(fptype InputX)
{
  int sign = 0;
  if (InputX < 0.0f) { InputX = -InputX; sign = 1; }

  fptype xInput = InputX;
  fptype expValues = expf(-0.5f * InputX * InputX);
  fptype xNPrimeofX = expValues * inv_sqrt_2xPI;

  fptype xK2 = 1.0f / (1.0f + 0.2316419f * xInput);
  fptype xK2_2 = xK2 * xK2;
  fptype xK2_3 = xK2_2 * xK2;
  fptype xK2_4 = xK2_3 * xK2;
  fptype xK2_5 = xK2_4 * xK2;

  fptype xLocal_1 = xK2 * 0.319381530f;
  fptype xLocal_2 = xK2_2 * (-0.356563782f) + xK2_3 * 1.781477937f;
  xLocal_2 += xK2_4 * (-1.821255978f);
  xLocal_2 += xK2_5 * 1.330274429f;

  fptype xLocal = (xLocal_2 + xLocal_1) * xNPrimeofX;
  xLocal = 1.0f - xLocal;

  fptype OutputX = xLocal;
  if (sign) OutputX = 1.0f - OutputX;
  return OutputX;
}

static void BlkSchlsEqEuroNoDiv_inline(
  const fptype *sptprice,
  const fptype *strike,
  const fptype *rate,
  const fptype *volatility,
  const fptype *time,
  const int *otype,
  fptype *OptionPrice,
  int size)
{
  for (int i = 0; i < size; i++) {
    fptype xStockPrice   = sptprice[i];
    fptype xStrikePrice  = strike[i];
    fptype xRiskFreeRate = rate[i];
    fptype xVolatility   = volatility[i];
    fptype xTime         = time[i];

    fptype xSqrtTime = sqrtf(xTime);
    fptype xLogTerm  = logf(xStockPrice / xStrikePrice);

    fptype xPowerTerm = 0.5f * xVolatility * xVolatility;

    fptype xD1 = (xRiskFreeRate + xPowerTerm) * xTime + xLogTerm;
    fptype xDen = xVolatility * xSqrtTime;
    xD1 = xD1 / xDen;
    fptype xD2 = xD1 - xDen;

    fptype NofXd1 = CNDF(xD1);
    fptype NofXd2 = CNDF(xD2);

    fptype FutureValueX = xStrikePrice * expf(-xRiskFreeRate * xTime);

    if (otype[i] == 0) { // CALL
      OptionPrice[i] = (xStockPrice * NofXd1) - (FutureValueX * NofXd2);
    } else {            // PUT
      fptype NegNofXd1 = 1.0f - NofXd1;
      fptype NegNofXd2 = 1.0f - NofXd2;
      OptionPrice[i] = (FutureValueX * NegNofXd2) - (xStockPrice * NegNofXd1);
    }
  }
}


/* Parallel task driver */
static void bs_run_tasks(fptype *prices, int BSIZE)
{
#pragma omp parallel
  {
#pragma omp single
    {
      for (int j = 0; j < NUM_RUNS; j++) {

        int i = 0;
        for (; i <= (numOptions_g - BSIZE); i += BSIZE) {
#pragma omp task firstprivate(i)
          {
            BlkSchlsEqEuroNoDiv_inline(
              &sptprice_g[i], &strike_g[i], &rate_g[i], &volatility_g[i],
              &otime_g[i], &otype_g[i], &prices[i], BSIZE
            );
          }
        }

        /* Remaining tail computed by creator thread */
        if (i < numOptions_g) {
          BlkSchlsEqEuroNoDiv_inline(
            &sptprice_g[i], &strike_g[i], &rate_g[i], &volatility_g[i],
            &otime_g[i], &otype_g[i], &prices[i], numOptions_g - i
          );
        }

#pragma omp taskwait

#ifdef ERR_CHK
        for (int k = 0; k < numOptions_g; k++) {
          fptype delta = data_g[k].DGrefval - prices[k];
          if (fabsf(delta) >= 1e-4f) numError_g++;
        }
#endif
      }
    }
  }
}

/* ---------------- Data generation ---------------- */

static void init_data(int numOptions)
{
  numOptions_g = numOptions;

  data_g = (OptionData*)malloc((size_t)numOptions_g * sizeof(OptionData));
  if (!data_g) { fprintf(stderr, "ERROR: malloc data_g failed\n"); exit(1); }

  /* SoA arrays */
  sptprice_g   = (fptype*)malloc((size_t)numOptions_g * sizeof(fptype));
  strike_g     = (fptype*)malloc((size_t)numOptions_g * sizeof(fptype));
  rate_g       = (fptype*)malloc((size_t)numOptions_g * sizeof(fptype));
  volatility_g = (fptype*)malloc((size_t)numOptions_g * sizeof(fptype));
  otime_g      = (fptype*)malloc((size_t)numOptions_g * sizeof(fptype));
  otype_g      = (int*)   malloc((size_t)numOptions_g * sizeof(int));

  if (!sptprice_g || !strike_g || !rate_g || !volatility_g || !otime_g || !otype_g) {
    fprintf(stderr, "ERROR: malloc SoA failed\n");
    exit(1);
  }

  /* Deterministic synthetic inputs */
  unsigned int rng = 123456789u;

  for (int i = 0; i < numOptions_g; i++) {
    /* Choose realistic ranges */
    fptype s     = 5.0f  + 25.0f * frand01(&rng);   // [5, 30]
    fptype k     = 1.0f  + 30.0f * frand01(&rng);   // [1, 31]
    fptype r     = 0.01f + 0.05f * frand01(&rng);   // [0.01, 0.06]
    fptype v     = 0.10f + 0.40f * frand01(&rng);   // [0.10, 0.50]
    fptype t     = 0.25f + 1.75f * frand01(&rng);   // [0.25, 2.0]
    int isPut    = (lcg_u32(&rng) & 1u) ? 1 : 0;

    data_g[i].s = s;
    data_g[i].strike = k;
    data_g[i].r = r;
    data_g[i].divq = 0.0f;
    data_g[i].v = v;
    data_g[i].t = t;
    data_g[i].OptionType = isPut ? 'P' : 'C';
    data_g[i].divs = 0.0f;
    data_g[i].DGrefval = 0.0f; // no reference in CAPBench mode

    sptprice_g[i]   = s;
    strike_g[i]     = k;
    rate_g[i]       = r;
    volatility_g[i] = v;
    otime_g[i]      = t;
    otype_g[i]      = isPut;
  }
}

static void free_data(void)
{
  free(data_g);
  free(sptprice_g);
  free(strike_g);
  free(rate_g);
  free(volatility_g);
  free(otime_g);
  free(otype_g);

  data_g = NULL;
  sptprice_g = strike_g = rate_g = volatility_g = otime_g = NULL;
  otype_g = NULL;
}

/* ---------------- main ---------------- */

int main(int argc, char **argv)
{
  options_t opt = parse_args(argc, argv);

  /* OpenMP setup */
  omp_set_dynamic(0);
  omp_set_num_threads(opt.nthreads);

  init_data(opt.numOptions);

  if (opt.blocksize > opt.numOptions) {
    fprintf(stderr, "ERROR: blocksize (%d) > numOptions (%d)\n", opt.blocksize, opt.numOptions);
    free_data();
    return 1;
  }

  fptype *prices = (fptype*)malloc((size_t)opt.numOptions * sizeof(fptype));
  if (!prices) { fprintf(stderr, "ERROR: malloc prices failed\n"); free_data(); return 1; }

  /* Time only the kernel region */
  struct timeval start, stop;
  gettimeofday(&start, NULL);

  bs_run_tasks(prices, opt.blocksize);

  gettimeofday(&stop, NULL);

  double elapsed = (double)(stop.tv_sec - start.tv_sec)
                 + (double)(stop.tv_usec - start.tv_usec) / 1e6;

  /* Your requested output */
  printf("  Runtime:       %f\n", elapsed);

#ifdef ERR_CHK
  printf("Num Errors: %d\n", numError_g);
#endif

  free(prices);
  free_data();
  return 0;
}
