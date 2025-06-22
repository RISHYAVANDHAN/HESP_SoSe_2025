#!/bin/bash
set -e

# Enable CUDA synchronous launches for better error reporting (useful when debugging)
export CUDA_LAUNCH_BLOCKING=1

INPUT_DIR="input"
OUTPUT_DIR="output"
BENCHMARK_DIR="benchmark"
BOX_SIZE=(10.0 10.0 10.0)  # Define box dimensions
PARTICLE_RADIUS=0.5        # Define particle radius

# Create required directories
mkdir -p "$OUTPUT_DIR"/{Simple,Particle_number,interesting_cases}
mkdir -p "$BENCHMARK_DIR"/{csv,plots}
mkdir -p "$INPUT_DIR/particles"

# Build the project
echo "[INFO] Building project..."
mkdir -p build
cd build
cmake .. && make -j
cd ..

echo -e "\n[INFO] Running Particle Number Benchmarks (Performance Analysis)"
PARTICLE_COUNTS=(100 200 300 400 500)

for N in "${PARTICLE_COUNTS[@]}"; do
  echo "[INFO] Generating input for $N particles..."
  python3 input/generate_input.py \
    --count $N \
    --box_size "${BOX_SIZE[@]}" \
    --radius $PARTICLE_RADIUS \
    --output "$INPUT_DIR/particles"
    
  INPUT_FILE="$INPUT_DIR/particles/particles_$N.txt"
  OUTPUT_PATH="$OUTPUT_DIR/Particle_number/$N"
  mkdir -p "$OUTPUT_PATH"
  
  echo "[INFO] Benchmarking $N particles..."
  $CMD_PREFIX ./build/molecular_dynamics \
    --input "$INPUT_FILE" \
    --steps 100 \
    --dt 0.001 \
    --box "${BOX_SIZE[@]}" \
    --particle_radius $PARTICLE_RADIUS \
    --stiffness 100.0 \
    --damping 0.1 \
    --bounce_coeff 0.8 \
    --output "$OUTPUT_PATH" \
    --freq 100 \
    --benchmark \
    --gravity 9.8
done

METHODS=("neighbour")
PARTICLE_COUNTS=(500)
RCUT=2.5  # In units of sigma

echo -e "\n[INFO] Running Performance Benchmarks"
for METHOD in "${METHODS[@]}"; do
    for N in "${PARTICLE_COUNTS[@]}"; do
        INPUT_FILE="$INPUT_DIR/particles/particles_$N.txt"
        OUTPUT_PATH="$OUTPUT_DIR/Performance/$METHOD/$N"
        BENCHMARK_PATH="$BENCHMARK_DIR/$METHOD"
        
        mkdir -p "$OUTPUT_PATH"
        mkdir -p "$BENCHMARK_PATH/csv"
        
        echo "[INFO] Running $N particles with $METHOD method..."
        $CMD_PREFIX ./build/molecular_dynamics \
            --input "$INPUT_FILE" \
            --steps 100 \
            --dt 0.00001 \
            --box "${BOX_SIZE[@]}" \
            --particle_radius $PARTICLE_RADIUS \
            --stiffness 100.0 \
            --damping 0.5 \
            --bounce_coeff 0.5 \
            --rcut $RCUT \
            --method "$METHOD" \
            --output "$OUTPUT_PATH" \
            --freq 100 \
            --benchmark \
            --gravity 9.8
    done
done

echo -e "\n[INFO] Collecting benchmark CSVs..."
find "$OUTPUT_DIR" -name "benchmark_*.csv" -exec cp {} "$BENCHMARK_DIR/csv/" \;

CSV_COUNT=$(find "$BENCHMARK_DIR/csv" -name "*.csv" | wc -l)
echo "[INFO] Collected $CSV_COUNT benchmark files"

if [ $CSV_COUNT -gt 0 ]; then
    echo -e "\n[INFO] Generating performance analysis plots..."
    python3 src/plot_benchmark.py "$BENCHMARK_DIR"/csv/*.csv --output_dir "$BENCHMARK_DIR/plots"
else
    echo "[WARNING] No benchmark files found for analysis"
fi

echo -e "\n[INFO] Performance Comparison Summary - Acceleration Techniques"
echo "======================================"

echo "Method | Particles | Mean Time (ms)"
echo "----------------------------------"
for METHOD in "${METHODS[@]}"; do
    for N in "${PARTICLE_COUNTS[@]}"; do
        CSV_FILE=$(find "$BENCHMARK_DIR/csv" -name "benchmark_${N}_*_$METHOD.csv" | head -1)
        if [ -f "$CSV_FILE" ]; then
            MEAN_TIME=$(awk -F, 'NR>1 {sum+=$2; count++} END {if(count>0) print sum/count; else print "N/A"}' "$CSV_FILE")
            printf "%-7s| %-10d| %.3f ms\n" "$METHOD" "$N" "$MEAN_TIME"
        else
            printf "%-7s| %-10d| %s\n" "$METHOD" "$N" "CSV not found"
        fi
    done
done

echo -e "\n[INFO] Done!"
echo "Output locations:"
echo "- Particle Number Tests:  $OUTPUT_DIR/Particle_number/"
echo "- Performance Tests:      $OUTPUT_DIR/Performance/"
echo "- Benchmark Data:         $BENCHMARK_DIR/csv/"
echo "- Performance Plots:      $BENCHMARK_DIR/plots/"
