#!/bin/bash

# Benchmark script for PARTH separator runtime analysis
# Usage: ./benchmark_sep_runtime.sh <folder_with_obj_files>

# Check if folder argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <folder_containing_obj_files>"
    echo "Example: $0 /Users/behrooz/Desktop/LastProject/Ahmed_big_meshes/"
    exit 1
fi

MESH_FOLDER="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARTH_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$PARTH_ROOT/output"

# Path to the PARTH_sep_runtime executable
EXECUTABLE="$PARTH_ROOT/cmake-build-release/dev/PARTH_sep_runtime"

# Check if executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: PARTH_sep_runtime executable not found at $EXECUTABLE"
    echo "Please build the project first."
    exit 1
fi

# Check if mesh folder exists
if [ ! -d "$MESH_FOLDER" ]; then
    echo "Error: Mesh folder does not exist: $MESH_FOLDER"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Convert to absolute paths
MESH_FOLDER=$(cd "$MESH_FOLDER" && pwd)
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

echo "Mesh folder: $MESH_FOLDER"
echo "Output folder: $OUTPUT_DIR"
echo "Executable: $EXECUTABLE"
echo ""

# Counter for processed files
processed_count=0
total_files=$(find "$MESH_FOLDER" -name "*.obj" | wc -l)

echo "Found $total_files .obj files to process"
echo ""

# Process each .obj file
for obj_file in "$MESH_FOLDER"/*.obj; do
    # Check if file exists (in case no .obj files found)
    if [ ! -f "$obj_file" ]; then
        echo "No .obj files found in $MESH_FOLDER"
        exit 1
    fi
    
    # Get absolute path of the obj file
    obj_file=$(realpath "$obj_file")
    filename=$(basename "$obj_file")
    
    echo "Processing: $filename"
    
    # Run with AMD permutation
    echo "  Running with AMD permutation..."
    "$EXECUTABLE" --input="$obj_file" --output="$OUTPUT_DIR" -p AMD
    if [ $? -ne 0 ]; then
        echo "  Error: Failed to run with AMD permutation for $filename"
    else
        echo "  AMD permutation completed successfully"
    fi
    
    # Run with METIS permutation
    echo "  Running with METIS permutation..."
    "$EXECUTABLE" --input="$obj_file" --output="$OUTPUT_DIR" -p METIS
    if [ $? -ne 0 ]; then
        echo "  Error: Failed to run with METIS permutation for $filename"
    else
        echo "  METIS permutation completed successfully"
    fi
    
    processed_count=$((processed_count + 1))
    echo "  Progress: $processed_count/$total_files files processed"
    echo ""
done

echo "Benchmark completed!"
echo "Total files processed: $processed_count"
echo "Results saved to: $OUTPUT_DIR/sep_runtime_analysis.csv"
