#!/bin/bash

# Simple script to run the quick_start executable
# Assumes the executable is already built
# Usage: ./run_quick_start.sh [--debug] [--input INPUT_MESH] [--output OUTPUT_DIR]

set -e  # Exit on any error

# Default to release build
BUILD_TYPE="release"
INPUT_MESH=""
OUTPUT_DIR=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug|-d)
            BUILD_TYPE="debug"
            shift
            ;;
        --input|-i)
            INPUT_MESH="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--debug] [--input INPUT_MESH] [--output OUTPUT_DIR]"
            echo "  --debug, -d            Use debug build (cmake-build-debug)"
            echo "  --input, -i INPUT      Specify input mesh file (optional)"
            echo "  --output, -o OUTPUT    Specify output directory (optional)"
            echo "  --help, -h             Show this help message"
            echo ""
            echo "Note: The quick_start program reads matrices from tests/matrices/"
            echo "      and doesn't require input/output arguments for basic operation."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Get the script directory (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get the project root directory (1 level up from scripts/)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Define paths based on build type
if [ "$BUILD_TYPE" = "debug" ]; then
    BUILD_DIR="cmake-build-debug"
    echo "🐛 Using DEBUG build"
else
    BUILD_DIR="cmake-build-release"
    echo "🚀 Using RELEASE build"
fi

EXECUTABLE="$PROJECT_ROOT/$BUILD_DIR/examples/PARTH_quick_start"
MATRICES_DIR="$PROJECT_ROOT/tests/matrices"

# Check if executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "❌ Error: Executable not found at $EXECUTABLE"
    echo "Please build the project first using:"
    echo "  ./scripts/build.sh"
    echo "Or compile just the examples with:"
    echo "  ./scripts/compile.sh"
    exit 1
fi

# Check if matrices directory exists
if [ ! -d "$MATRICES_DIR" ]; then
    echo "❌ Error: Matrices directory not found: $MATRICES_DIR"
    echo "Please generate matrices first using:"
    echo "  ./scripts/tests/create_matrices.sh"
    exit 1
fi

# Check if required matrices exist
ORIGINAL_MATRIX="$MATRICES_DIR/original_matrix.mtx"
MATRIX_1="$MATRICES_DIR/1_matrix.mtx"

if [ ! -f "$ORIGINAL_MATRIX" ]; then
    echo "❌ Error: Original matrix not found: $ORIGINAL_MATRIX"
    echo "Please generate matrices first using:"
    echo "  ./scripts/tests/create_matrices.sh"
    exit 1
fi

if [ ! -f "$MATRIX_1" ]; then
    echo "❌ Error: 1_matrix.mtx not found: $MATRIX_1"
    echo "Please generate matrices first using:"
    echo "  ./scripts/tests/create_matrices.sh"
    exit 1
fi

echo "Running PARTH quick_start..."
echo "Executable: $EXECUTABLE"
echo "Matrices directory: $MATRICES_DIR"

# Change to project root directory so relative paths work correctly
cd "$PROJECT_ROOT"

# Build command arguments
ARGS=()
if [ ! -z "$INPUT_MESH" ]; then
    ARGS+=("-i" "$INPUT_MESH")
fi
if [ ! -z "$OUTPUT_DIR" ]; then
    ARGS+=("-o" "$OUTPUT_DIR")
fi

# Run the executable
echo "🏃 Executing quick_start program..."
echo "----------------------------------------"
"$EXECUTABLE" "${ARGS[@]}"
echo "----------------------------------------"

echo "✅ Done! Quick start execution completed successfully."
