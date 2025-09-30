/*
 * stencil.h — Space-invariant linear symmetric stencil (CAPBench style)
 *
 * Applies a STAR (4*R+1 points) or COMPACT ((2R+1)^2) filter to
 * an n×n grid for (iterations+1) iterations (first is warm-up).
 * In each iteration: OUT accumulates contributions; IN receives +1.0 to
 * force reuse/neighborhood update (as in PRK).
 *
 * Return:
 *   - Total time in seconds (measured from iter=1..iterations).
 *   - L1 norm (average in interior) via *l1_norm_out (for verification).
 */
#ifndef STENCIL_H
#define STENCIL_H

#ifdef __cplusplus
extern "C"
{
#endif

    typedef enum
    {
        STENCIL_STAR = 0,
        STENCIL_COMPACT = 1
    } stencil_type_t;

    /* Executes the kernel. Returns measured seconds. */
    double stencil_kernel(int n, int radius, int iterations,
                          stencil_type_t type,
                          double *l1_norm_out);

#ifdef __cplusplus
}
#endif

#endif /* STENCIL_H */
