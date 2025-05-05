# Output file for results
OUTPUT_FILE="bandwidth_results.csv"

# Buffer sizes to test (in number of elements)
BUFFER_SIZES=(1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216)

# Number of warm-up and measurement iterations
NWARM=2
NIT=10

# Paths to executables
SERIAL_EXEC="../../build/stream/stream-base"
PARALLEL_EXEC="../../build/stream/stream-omp-host"
CUDA_EXEC="../../build/stream/stream-cuda"

# Check if executables exist
if [[ ! -f "$SERIAL_EXEC" || ! -f "$PARALLEL_EXEC" || ! -f "$CUDA_EXEC" ]]; then
    echo "Error: One or more executables not found. Please build the project first using 'make all'."
    exit 1
fi

# Write CSV header
echo "BufferSize,Version,ElapsedTime(ms),Bandwidth(MLUP/s),Bandwidth(GB/s)" > "$OUTPUT_FILE"

# Function to run a benchmark and extract results
run_benchmark() {
    local exec=$1
    local buffer_size=$2
    local version=$3

    # Run the benchmark and capture output
    output=$($exec $buffer_size $NWARM $NIT)

    # Extract relevant metrics from the output
    elapsed_time=$(echo "$output" | grep "elapsed time" | awk '{print $3}')
    mlups=$(echo "$output" | grep "MLUP/s" | awk '{print $2}')
    bandwidth=$(echo "$output" | grep "bandwidth" | awk '{print $2}')

    # Write results to CSV
    echo "$buffer_size,$version,$elapsed_time,$mlups,$bandwidth" >> "$OUTPUT_FILE"
}

# Run benchmarks for each buffer size
for buffer_size in "${BUFFER_SIZES[@]}"; do
    echo "Running benchmarks for buffer size: $buffer_size"

    # Serial version
    run_benchmark "$SERIAL_EXEC" "$buffer_size" "Serial"

    # Parallel version
    run_benchmark "$PARALLEL_EXEC" "$buffer_size" "Parallel"

    # CUDA version
    run_benchmark "$CUDA_EXEC" "$buffer_size" "CUDA"
done

echo "Evaluation complete. Results saved to $OUTPUT_FILE."