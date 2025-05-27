#
# Copyright(C) 2025 Eduardo Lemos Paschoalini
#

# Directories
export ROOTDIR=$PWD
export RESULTSDIR=$ROOTDIR/results
export BINDIR=$ROOTDIR/x86/bin
export ITERATIONS=2

mkdir -p $RESULTSDIR
# gf is km lu tsp
# fn gf is km lu
# tiny small standard large huge

for kernel in fast fn; do
    echo "running $kernel"
    mkdir -p $RESULTSDIR/$kernel

    # Weak scaling
    i=0
    while [ $i -lt $ITERATIONS ]; do
        echo "running weak scaling test: iteration $i"
        for nprocs in 1 4 8 16; do
            for class in standard; do
                for regalloc_algo in fast basic greedy pbqp; do
                    out_file="$RESULTSDIR/$kernel/${class}-${regalloc_algo}-${nprocs}.txt"
                    bin="$BINDIR/$kernel/$regalloc_algo.intel"

                    echo ">> Iteration $i" >> "$out_file"
                    echo "[COMMAND] $bin --verbose --class $class --nthreads $nprocs" >> "$out_file"

                    $bin --verbose --class "$class" --nthreads "$nprocs" >> "$out_file" 2>&1
                    echo -e "\n---\n" >> "$out_file"
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
