#
# Copyright(C) 2025 Eduardo Lemos Paschoalini
#

# Directories
export ROOTDIR=$PWD
export RESULTSDIR=$ROOTDIR/results
export BINDIR=$ROOTDIR/x86/bin
export ITERATIONS=10

mkdir -p $RESULTSDIR
#gf is km lu tsp

for kernel in fast fn gf; do
    echo "running $kernel"
    mkdir -p $RESULTSDIR/$kernel

    # Weak scaling
    i=0
    while [ $i -lt $ITERATIONS ]; do
        echo "running weak scaling test: iteration $i"
        for nprocs in 1 2 4 8 16; do
            for class in tiny small; do
                for regalloc_algo in fast basic greedy pbqp; do
                    $BINDIR/$kernel/$regalloc_algo.intel --verbose --class $class --nthreads $nprocs >> $RESULTSDIR/$kernel/$class-$regalloc_algo-$nprocs.txt
                done
            done
        done
        i=$((i + 1))
    done
    
    # Strong scaling
    # i=0
    # while [ $i -lt $ITERATIONS ]; do
    #     echo "running strong scaling test: iteration $i"
    #     for nprocs in 3 5 6 7 9 10 11 12 13 14 15; do
    #         $BINDIR/$kernel.intel --verbose --class standard --nthreads $nprocs &>> $RESULTSDIR/$kernel-standard-$nprocs.intel
    #     done
    #     i=$((i + 1))
    # done
done
