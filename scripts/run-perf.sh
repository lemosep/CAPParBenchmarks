#
# Copyright(C) 2025 Eduardo Lemos Paschoalini
#

# Directories
export ROOTDIR=$PWD
export RESULTSDIR=$ROOTDIR/results_perf
export BINDIR=$ROOTDIR/x86/bin

mkdir -p $RESULTSDIR

# Kernels to test
kernels="fast fn gf is km lu tsp rt nb bc dgemm sp sten"
nprocs_list="1 4 6 8 12 16"
regalloc_algos="fast basic greedy pbqp"
class="standard"

for kernel in $kernels; do
    echo "running perf stat for $kernel"
    mkdir -p $RESULTSDIR/$kernel

    for nprocs in $nprocs_list; do
        mkdir -p $RESULTSDIR/$kernel/${nprocs}-threads

        for regalloc_algo in $regalloc_algos; do
            bin="$BINDIR/$kernel/$regalloc_algo.intel"
            out_file="$RESULTSDIR/$kernel/${nprocs}-threads/${regalloc_algo}_perf.txt"

            echo "[PERF COMMAND] $bin --class $class --nthreads $nprocs"
            perf stat -o "$out_file" -e \
                cycles,instructions,cache-references,cache-misses,LLC-loads,LLC-load-misses \
                -- $bin --class "$class" --nthreads "$nprocs"
        done
    done
done