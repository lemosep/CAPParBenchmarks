#
# Copyright(C) 2025 Eduardo Lemos Paschoalini
#

# Directories
export ROOTDIR=$PWD
export RESULTSDIR=$ROOTDIR/results
export BINDIR=$ROOTDIR/x86/bin
export ITERATIONS=2

mkdir -p $RESULTSDIR

# tiny small standard large huge

for kernel in fast fn gf is km lu tsp rt nb bc dgemm sp sten; do
    echo "running $kernel"
    mkdir -p $RESULTSDIR/$kernel

    # Weak scaling
    i=0
    while [ $i -lt $ITERATIONS ]; do
        echo "running weak scaling test: iteration $i"
        for nprocs in 1 4 6 8 12 16; do
            mkdir -p $RESULTSDIR/$kernel/${nprocs}-threads
            for class in standard; do
                for regalloc_algo in fast basic greedy pbqp; do    
                    out_file="$RESULTSDIR/$kernel/${nprocs}-threads/${regalloc_algo}.txt"
                    bin="$BINDIR/$kernel/$regalloc_algo.intel"

                    echo ">> Iteration $i" >> "$out_file"
                    echo "[COMMAND] $bin --class $class --nthreads $nprocs" >> "$out_file"

                    $bin --class "$class" --nthreads "$nprocs" >> "$out_file" 2>&1
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
