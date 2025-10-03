#!/bin/bash

# Script to compare Rust vs Python Helmholtz 3D performance
# This script runs both implementations and presents a side-by-side comparison

echo "========================================="
echo "Helmholtz 3D Performance Comparison"
echo "Rust vs Python Implementation"
echo "========================================="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    echo -e "${RED}Error: Must run from the helmotz project root directory${NC}"
    exit 1
fi

# Build Rust version in release mode
echo "Building Rust benchmark (release mode)..."
cargo build --release --example benchmark_3d 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Warning: Failed to build Rust benchmark${NC}"
    echo "Trying with verbose output:"
    cargo build --release --example benchmark_3d
fi

echo
echo "========================================="
echo -e "${GREEN}RUST IMPLEMENTATION${NC}"
echo "========================================="

# Run Rust benchmark
RUST_OUTPUT=$(mktemp)
if cargo run --release --example benchmark_3d 2>/dev/null > "$RUST_OUTPUT"; then
    cat "$RUST_OUTPUT"
    
    # Extract key metrics for comparison
    RUST_32_TIME=$(grep "32³" "$RUST_OUTPUT" | awk '{print $4}')
    RUST_32_ITERS=$(grep "32³" "$RUST_OUTPUT" | awk '{print $6}')
    RUST_32_THROUGHPUT=$(grep "32³" "$RUST_OUTPUT" | awk '{print $10}')
else
    echo -e "${RED}Failed to run Rust benchmark${NC}"
    RUST_32_TIME="N/A"
    RUST_32_ITERS="N/A"
    RUST_32_THROUGHPUT="N/A"
fi

echo
echo "========================================="
echo -e "${GREEN}PYTHON IMPLEMENTATION${NC}"
echo "========================================="

# Check Python environment
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}Python not found${NC}"
    exit 1
fi

# Run Python benchmark
PYTHON_OUTPUT=$(mktemp)
if $PYTHON_CMD wavesim_py/examples/benchmark_3d.py 2>/dev/null > "$PYTHON_OUTPUT"; then
    cat "$PYTHON_OUTPUT"
    
    # Extract key metrics for comparison
    PYTHON_32_TIME=$(grep "32³" "$PYTHON_OUTPUT" | awk '{print $4}')
    PYTHON_32_ITERS=$(grep "32³" "$PYTHON_OUTPUT" | awk '{print $6}')
    PYTHON_32_THROUGHPUT=$(grep "32³" "$PYTHON_OUTPUT" | awk '{print $10}')
else
    echo -e "${YELLOW}Warning: Python benchmark failed or not available${NC}"
    echo "Attempting to run with error output:"
    $PYTHON_CMD wavesim_py/examples/benchmark_3d.py
    PYTHON_32_TIME="N/A"
    PYTHON_32_ITERS="N/A"
    PYTHON_32_THROUGHPUT="N/A"
fi

echo
echo "========================================="
echo -e "${GREEN}PERFORMANCE COMPARISON SUMMARY${NC}"
echo "========================================="
echo
echo "32x32x32 Grid Comparison:"
echo "-----------------------------------------"
echo "                | Rust        | Python"
echo "----------------|-------------|------------"
printf "Time (s)        | %-11s | %-11s\n" "$RUST_32_TIME" "$PYTHON_32_TIME"
printf "Iterations      | %-11s | %-11s\n" "$RUST_32_ITERS" "$PYTHON_32_ITERS"
printf "Mvox-it/s       | %-11s | %-11s\n" "$RUST_32_THROUGHPUT" "$PYTHON_32_THROUGHPUT"

# Calculate speedup if both values are available
if [[ "$RUST_32_TIME" != "N/A" && "$PYTHON_32_TIME" != "N/A" ]]; then
    # Use bc for floating point calculation
    if command -v bc &> /dev/null; then
        SPEEDUP=$(echo "scale=2; $PYTHON_32_TIME / $RUST_32_TIME" | bc 2>/dev/null)
        if [ $? -eq 0 ]; then
            echo
            echo -e "${YELLOW}Rust speedup: ${SPEEDUP}x faster${NC}"
        fi
    fi
fi

echo
echo "========================================="
echo "Note: Both implementations use the same:"
echo "  - Problem size and geometry"
echo "  - Convergence criteria (1e-6)"
echo "  - Relaxation parameter (α=0.75)"
echo "  - Boundary conditions (periodic)"
echo "========================================="

# Cleanup
rm -f "$RUST_OUTPUT" "$PYTHON_OUTPUT"