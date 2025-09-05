#!/bin/bash

# Simple script to run the example_creator executable on fish.obj
# Assumes the executable is already built
# Usage: ./create_matrices.sh [--debug]

set -e  # Exit on any error

# Default to release build
BUILD_TYPE="release"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug|-d)
            BUILD_TYPE="debug"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--debug]"
            echo "  --debug, -d    Use debug build (cmake-build-debug)"
            echo "  --help, -h     Show this help message"
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

# Get the project root directory (2 levels up from scripts/tests/)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Define paths based on build type
if [ "$BUILD_TYPE" = "debug" ]; then
    BUILD_DIR="cmake-build-debug"
    echo "🐛 Using DEBUG build"
else
    BUILD_DIR="cmake-build-release"
    echo "🚀 Using RELEASE build"
fi

EXECUTABLE="$PROJECT_ROOT/$BUILD_DIR/tests/PARTH_example_creator"
INPUT_MESH="$PROJECT_ROOT/tests/input/fish.obj"
OUTPUT_DIR="$PROJECT_ROOT/tests/matrices"

# Check if executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "❌ Error: Executable not found at $EXECUTABLE"
    echo "Please build the project first!"
    exit 1
fi

# Check if input mesh exists
if [ ! -f "$INPUT_MESH" ]; then
    echo "❌ Error: Input mesh file not found: $INPUT_MESH"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Running example_creator on fish.obj..."
echo "Input: $INPUT_MESH"
echo "Output: $OUTPUT_DIR"

# Run the executable
"$EXECUTABLE" -i "$INPUT_MESH" -o "$OUTPUT_DIR"

echo "✅ Done! Generated files:"
ls -la "$OUTPUT_DIR"/fish* 2>/dev/null || echo "No files generated"
