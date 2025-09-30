/*
 * branch.h - Branching benchmark kernel (CAPBench style).
 *
 * Adaptation of PRK "Branching Bonanza" to CAPBench standard.
 */

#ifndef _BRANCH_H_
#define _BRANCH_H_

/* Branch types */
typedef enum
{
    BR_VECTOR_STOP = 66,
    BR_VECTOR_GO = 77,
    BR_NO_VECTOR = 88,
    BR_INS_HEAVY = 99 /* (not supported in this adaptation) */
} branch_type_t;

/*
 * Executes the kernel with branches.
 * Returns the final vector sum (checksum).
 * nfunc/rank are relevant only for INS_HEAVY (not supported here).
 */
int branch_with_branches(int vector_length, int iterations,
                         branch_type_t btype, int *nfunc, int *rank);

/*
 * Executes the kernel without branches (equivalent path).
 * Returns the final vector sum (checksum).
 */
int branch_without_branches(int vector_length, int iterations,
                            branch_type_t btype, int *nfunc, int *rank);

#endif /* _BRANCH_H_ */
