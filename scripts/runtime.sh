#!/usr/bin/env bash
#
# Copyright(C) 2025 Eduardo Lemos Paschoalini
#

set -euo pipefail

# Directories
export ROOTDIR=$PWD
export RESULTSDIR="$ROOTDIR/results"
export BINDIR="$ROOTDIR/x86/bin"
export ITERATIONS=10
export CLASS="tiny"   # problema fixo -> isso é strong scaling, não weak

mkdir -p "$RESULTSDIR"

# CSV unificado de runtime
RUNTIME_CSV="$RESULTSDIR/runtime_raw.csv"
echo "kernel,allocator,threads,run_id,runtime_ms" > "$RUNTIME_CSV"

# tiny small standard large huge

for kernel in fast fn gf is km lu tsp rt nb bc dgemm sp sten gemm; do
    echo "running $kernel"

    for (( run_id=1; run_id<=ITERATIONS; run_id++ )); do
        echo "  iteration $run_id"

        for nprocs in 1 4 6 8 12 16; do
            out_dir="$RESULTSDIR/$kernel/${nprocs}-threads"
            mkdir -p "$out_dir"

            for regalloc_algo in basic greedy pbqp; do
                out_file="$out_dir/${regalloc_algo}.txt"
                bin="$BINDIR/$kernel/$regalloc_algo.intel"

                if [ ! -x "$bin" ]; then
                    echo "WARNING: binary not found or not executable: $bin" >&2
                    continue
                fi

                echo "    [$kernel] $regalloc_algo, ${nprocs} threads (run $run_id)"

                # comando a ser executado
                cmd="$bin --class $CLASS --nthreads $nprocs"

                # log “bonitinho”
                {
                    echo ">> Iteration $run_id"
                    echo "[COMMAND] $cmd"
                } >> "$out_file"

                              output=$($cmd 2>&1)
                echo "$output" >> "$out_file"
                echo -e "\n---\n" >> "$out_file"

                # === EXTRAÇÃO DO TEMPO DE EXECUÇÃO ===
                runtime_s=$(printf '%s\n' "$output" | \
                    awk '/Runtime:/ { print $2; exit }')

                if [ -z "${runtime_s:-}" ]; then
                    echo "WARNING: could not extract runtime for $kernel $regalloc_algo ${nprocs} threads (run $run_id)" >&2
                else
                    runtime_ms=$(awk -v v="$runtime_s" 'BEGIN { printf "%.6f", v * 1000.0 }')
                    echo "$kernel,$regalloc_algo,$nprocs,$run_id,$runtime_ms" >> "$RUNTIME_CSV"
                fi


            done  # regalloc_algo
        done      # nprocs
    done          # run_id
done              # kernel
