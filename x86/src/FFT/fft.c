/* Fast Fourier Transform */

#include <global.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <getopt.h>
#include <string.h>

/* Prototypes */
int main(int argc, char **argv);
void ccopy(int n, double x[], double y[]);
void cfft2(int n, double x[], double y[], double w[], double sgn);
void cffti(int n, double w[]);
double ggl(double *ds);
void step(int n, int mj, double a[], double b[], double c[], double d[],
          double w[], double sgn);
void timestamp(void);


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
} options_t;

static void usage(const char *prog) {
  fprintf(stderr,
          "Usage: %s [--nthreads N] [--class tiny|small|standard|large|huge]\n"
          "Defaults: --nthreads=omp_get_max_threads(), --class=standard\n",
          prog);
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

static options_t parse_args(int argc, char **argv) {
  options_t opt;
  opt.nthreads = -1;
  opt.cls = CLASS_STANDARD;

  static struct option long_opts[] = {
      {"nthreads", required_argument, 0, 't'},
      {"class",    required_argument, 0, 'c'},
      {"help",     no_argument,       0, 'h'},
      {0,0,0,0}
  };

  int c;
  while ((c = getopt_long(argc, argv, "t:c:h", long_opts, NULL)) != -1) {
    switch (c) {
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
      case 'h':
      default:
        usage(argv[0]);
        exit(0);
    }
  }

  if (opt.nthreads < 0) opt.nthreads = omp_get_max_threads();
  return opt;
}

static int pow2_int(int ln2) {
  return 1 << ln2;
}

static void class_params(bench_class_t cls, int *out_ln2, int *out_nits) {
  switch (cls) {
    case CLASS_TINY:     *out_ln2 = 10; *out_nits = 2000; break; // N=1024
    case CLASS_SMALL:    *out_ln2 = 12; *out_nits = 1000; break; // N=4096
    case CLASS_STANDARD: *out_ln2 = 16; *out_nits = 200;  break; // N=65536
    case CLASS_LARGE:    *out_ln2 = 20; *out_nits = 45;   break; // N=262144
    case CLASS_HUGE:     *out_ln2 = 22; *out_nits = 10;   break; // N=1048576
    default:             *out_ln2 = 16; *out_nits = 200;  break;
  }
}

/* ---------------- main ---------------- */

int main(int argc, char **argv)
{
  options_t opt = parse_args(argc, argv);

  omp_set_num_threads(opt.nthreads);

  int ln2, nits;
  class_params(opt.cls, &ln2, &nits);
  int n = pow2_int(ln2);

  /* Allocate */
  double *w = (double*)malloc((size_t)n * sizeof(double));
  double *x = (double*)malloc((size_t)2 * (size_t)n * sizeof(double));
  double *y = (double*)malloc((size_t)2 * (size_t)n * sizeof(double));
  double *z = (double*)malloc((size_t)2 * (size_t)n * sizeof(double));

  if (!w || !x || !y || !z) {
    fprintf(stderr, "ERROR: allocation failed for N=%d\n", n);
    free(w); free(x); free(y); free(z);
    return 1;
  }

  /* Init data */
  static double seed = 331.0;
  for (int i = 0; i < 2 * n; i += 2) {
    double z0 = ggl(&seed);
    double z1 = ggl(&seed);
    x[i]   = z0;  z[i]   = z0;
    x[i+1] = z1;  z[i+1] = z1;
  }

  /* Precompute twiddles */
  cffti(n, w);

  /* Warm-up / verify once (optional but helps catch obvious issues) */
  {
    double sgn = +1.0;
    cfft2(n, x, y, w, sgn);
    sgn = -1.0;
    cfft2(n, y, x, w, sgn);

    /* Restore x from z so timed region starts from same input */
    for (int i = 0; i < 2 * n; i += 2) {
      x[i]   = z[i];
      x[i+1] = z[i+1];
    }
  }

  /* Timing: nits forward+backward */
  double wtime = omp_get_wtime();
  for (int it = 0; it < nits; it++) {
    double sgn = +1.0;
    cfft2(n, x, y, w, sgn);
    sgn = -1.0;
    cfft2(n, y, x, w, sgn);
  }
  wtime = omp_get_wtime() - wtime;

  /* Your requested output */
  printf("  Runtime:       %f\n", wtime);

  free(w);
  free(x);
  free(y);
  free(z);

  return 0;
}

/* ---------------- Original FFT code (unchanged) ---------------- */

void ccopy(int n, double x[], double y[])
{
  for (int i = 0; i < n; i++) {
    y[i*2+0] = x[i*2+0];
    y[i*2+1] = x[i*2+1];
  }
}

void cfft2(int n, double x[], double y[], double w[], double sgn)
{
  int m  = (int)(log((double)n) / log(1.99));
  int mj = 1;

  int tgle = 1;

  step(n, mj, &x[0*2+0], &x[(n/2)*2+0], &y[0*2+0], &y[mj*2+0], w, sgn);

  if (n == 2) {
    return;
  }

  for (int j = 0; j < m - 2; j++) {
    mj = mj * 2;
    if (tgle) {
      step(n, mj, &y[0*2+0], &y[(n/2)*2+0], &x[0*2+0], &x[mj*2+0], w, sgn);
      tgle = 0;
    } else {
      step(n, mj, &x[0*2+0], &x[(n/2)*2+0], &y[0*2+0], &y[mj*2+0], w, sgn);
      tgle = 1;
    }
  }

  if (tgle) {
    ccopy(n, y, x);
  }

  mj = n / 2;
  step(n, mj, &x[0*2+0], &x[(n/2)*2+0], &y[0*2+0], &y[mj*2+0], w, sgn);
}

void cffti ( int n, double w[] )
{
  double arg;
  double aw;
  int i;
  int n2;
  const double pi = 3.141592653589793;

  n2 = n / 2;
  aw = 2.0 * pi / ( ( double ) n );

# pragma omp parallel \
    shared ( aw, n, w ) \
    private ( arg, i )

# pragma omp for nowait
  for ( i = 0; i < n2; i++ )
  {
    arg = aw * ( ( double ) i );
    w[i*2+0] = cos ( arg );
    w[i*2+1] = sin ( arg );
  }
}

double ggl(double *seed)
{
  double d2 = 0.2147483647e10;
  double t = (double)(*seed);
  t = fmod(16807.0 * t, d2);
  *seed = (double)t;
  return (double)((t - 1.0) / (d2 - 1.0));
}

void step(int n, int mj, double a[], double b[], double c[], double d[],
          double w[], double sgn)
{
  int mj2 = 2 * mj;
  int lj  = n / mj2;

#pragma omp parallel shared(a, b, c, d, lj, mj, mj2, sgn, w)
  {
    double ambr, ambu;
    double wjw[2];

#pragma omp for nowait
    for (int j = 0; j < lj; j++) {
      int jw = j * mj;
      int ja = jw;
      int jb = ja;
      int jc = j * mj2;
      int jd = jc;

      wjw[0] = w[jw*2+0];
      wjw[1] = w[jw*2+1];

      if (sgn < 0.0) {
        wjw[1] = -wjw[1];
      }

      for (int k = 0; k < mj; k++) {
        c[(jc+k)*2+0] = a[(ja+k)*2+0] + b[(jb+k)*2+0];
        c[(jc+k)*2+1] = a[(ja+k)*2+1] + b[(jb+k)*2+1];

        ambr = a[(ja+k)*2+0] - b[(jb+k)*2+0];
        ambu = a[(ja+k)*2+1] - b[(jb+k)*2+1];

        d[(jd+k)*2+0] = wjw[0] * ambr - wjw[1] * ambu;
        d[(jd+k)*2+1] = wjw[1] * ambr + wjw[0] * ambu;
      }
    }
  }
}

void timestamp(void)
{
#define TIME_SIZE 40
  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  time_t now = time(NULL);
  tm = localtime(&now);
  strftime(time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm);
  printf("%s\n", time_buffer);
#undef TIME_SIZE
}
