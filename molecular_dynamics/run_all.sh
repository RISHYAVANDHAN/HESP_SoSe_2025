#!/bin/bash
set -e

INPUT_DIR="input"
OUTPUT_DIR="output"
BENCHMARK_DIR="benchmark"

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

:<<'End comment'
echo -e "\n[INFO] Running Simple Test Cases (stable, repulsive, attractive)"
for TEST_CASE in stable repulsive attractive; do
  OUTPUT_PATH="$OUTPUT_DIR/Simple/$TEST_CASE"
  mkdir -p "$OUTPUT_PATH"
  echo "[INFO] Running $TEST_CASE test case..."
  ./build/molecular_dynamics \
    --test "$TEST_CASE" \
    --sigma 1.0 \
    --epsilon 0.35 \
    --dt 0.01 \
    --steps 100 \
    --output "$OUTPUT_PATH" \
    --freq 100\
    --benchmark
done

echo -e "\n[INFO] Generating input files..."
python3 input/generate_input.py --case collision_clusters --count_per_cluster 100 --output "$INPUT_DIR/particles"
python3 input/generate_input.py --case repulsive_shell --points_on_shell 200 --output "$INPUT_DIR/particles"
python3 input/generate_input.py --case attractive_core --cluster_count 300 --output "$INPUT_DIR/particles"

echo -e "\n[INFO] Running Particle Number Benchmarks (Performance Analysis)"
PARTICLE_COUNTS=(10 25 50 150 250 500)

for N in "${PARTICLE_COUNTS[@]}"; do
  echo "[INFO] Generating input for $N particles..."
  python3 input/generate_input.py --counts $N --output "$INPUT_DIR/particles"
  INPUT_FILE="$INPUT_DIR/particles/particles_$N.txt"
  OUTPUT_PATH="$OUTPUT_DIR/Particle_number/$N"
  mkdir -p "$OUTPUT_PATH"
  echo "[INFO] Benchmarking $N particles..."
  
  # Run longer simulation for better statistics
  ./build/molecular_dynamics \
    --input "$INPUT_FILE" \
    --steps 1000 \
    --dt 0.001 \
    --sigma 0.0501 \
    --epsilon 0.015 \
    --output "$OUTPUT_PATH" \
    --freq 100\
    --benchmark
done

echo -e "\n[INFO] Running Interesting Cases"
declare -A INTERESTING_CASE_PATTERNS=(
    ["collision_clusters"]="particles_collision_clusters_*"
    ["repulsive_shell"]="particles_repulsive_shell_*"
    ["attractive_core"]="particles_attractive_core_*"
)

for CASE_NAME in "${!INTERESTING_CASE_PATTERNS[@]}"; do
    # Find the actual generated file
    INPUT_FILES=("$INPUT_DIR/particles/"${INTERESTING_CASE_PATTERNS[$CASE_NAME]})
    if [ ${#INPUT_FILES[@]} -eq 0 ]; then
        echo "Error: No input files found for $CASE_NAME"
        exit 1
    fi
    INPUT_FILE=${INPUT_FILES[0]}  # Take first matching file
    
    OUTPUT_PATH="$OUTPUT_DIR/interesting_cases/$CASE_NAME"
    mkdir -p "$OUTPUT_PATH"
    echo "[INFO] Running $CASE_NAME with input file: $(basename $INPUT_FILE)"
    
    ./build/molecular_dynamics \
        --input "$INPUT_FILE" \
        --dt 0.001 \
        --steps 1000 \
        --sigma 0.0501 \
        --epsilon 0.015 \
        --output "$OUTPUT_PATH" \
        --freq 100\
        --benchmark
done
End comment

METHODS=("base" "cutoff" "cell" "neighbour")
PARTICLE_COUNTS=(10 25 50 150 250 500)
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
        
        ./build/molecular_dynamics \
            --input "$INPUT_FILE" \
            --steps 1000 \
            --dt 0.001 \
            --sigma 0.0501 \
            --epsilon 0.015 \
            --rcut 2.5 \
            --method "$METHOD" \
            --box 10.0 10.0 10.0 \
            --output "$OUTPUT_PATH" \
            --freq 100 \
            --benchmark
    done
done

echo -e "\n[INFO] Collecting benchmark CSVs..."
# Copy all benchmark CSVs to central location
find "$OUTPUT_DIR" -name "benchmark_*.csv" -exec cp {} "$BENCHMARK_DIR/csv/" \;

# Count collected files
CSV_COUNT=$(find "$BENCHMARK_DIR/csv" -name "*.csv" | wc -l)
echo "[INFO] Collected $CSV_COUNT benchmark files"

if [ $CSV_COUNT -gt 0 ]; then
    echo -e "\n[INFO] Generating performance analysis plots..."
    python3 src/plot_benchmark.py "$BENCHMARK_DIR"/csv/*.csv --output_dir "$BENCHMARK_DIR/plots"
    
    echo -e "\n[INFO] Generating scaling analysis only..."
    python3 src/plot_benchmark.py "$BENCHMARK_DIR"/csv/benchmark_*_*.csv --output_dir "$BENCHMARK_DIR/plots" --scaling-only
else
    echo "[WARNING] No benchmark files found for analysis"
fi

:<< 'End comment'
echo -e "\n[INFO] Performance Analysis Summary - Implementation of MD"
echo "======================================"

# Show particle count vs performance summary
echo "Particle Count | Mean Time per Step"
echo "------------------------------------"
for csv_file in "$BENCHMARK_DIR"/csv/benchmark_*.csv; do
    if [ -f "$csv_file" ]; then
        # Extract particle count from filename
        PARTICLES=$(basename "$csv_file" | sed 's/benchmark_\([0-9]*\)_.*/\1/')
        
        # Calculate mean time from CSV (skip header)
        if [ -f "$csv_file" ] && [ -s "$csv_file" ]; then
            MEAN_TIME=$(tail -n +2 "$csv_file" | cut -d',' -f2 | awk '{sum+=$1; count++} END {if(count>0) printf "%.3f", sum/count; else print "N/A"}')
            printf "%13s | %s ms\n" "$PARTICLES" "$MEAN_TIME"
        fi
    fi
done
End comment

echo -e "\n[INFO] Performance Comparison Summary - Accelaration Techniques"
echo "======================================"

echo "Method | Particles | Mean Time (ms)"
echo "----------------------------------"
for METHOD in "${METHODS[@]}"; do
    for N in "${PARTICLE_COUNTS[@]}"; do
        CSV_FILE="$BENCHMARK_DIR/$METHOD/csv/benchmark_${N}_*.csv"
        if [ -f $CSV_FILE ]; then
            MEAN_TIME=$(awk -F, 'NR>1 {sum+=$2; count++} END {print sum/count}' $CSV_FILE)
            printf "%-7s| %-10d| %.3f ms\n" "$METHOD" "$N" "$MEAN_TIME"
        fi
    done
done

echo -e "\n[INFO] Done!"
echo "Output locations:"
echo "- Simple Test Cases:      $OUTPUT_DIR/Simple/"
echo "- Particle Number Tests:  $OUTPUT_DIR/Particle_number/"
echo "- Interesting Cases:      $OUTPUT_DIR/interesting_cases/"
echo "- Benchmark Data:         $BENCHMARK_DIR/csv/"
echo "- Performance Plots:      $BENCHMARK_DIR/plots/"

# Display key results
if [ -f "$BENCHMARK_DIR/plots/performance_scaling.png" ]; then
    echo "- Scaling Analysis:       $BENCHMARK_DIR/plots/performance_scaling.png"
fi