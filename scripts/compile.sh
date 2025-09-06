#!/bin/bash

# Simple compile script for example_creator
# Compiles just the test executable without full project rebuild
# Usage: ./compile.sh [--debug]

set -e  # Exit on any error

# Default to release build
BUILD_TYPE="Release"
BUILD_DIR_NAME="cmake-build-release"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug|-d)
            BUILD_TYPE="Debug"
            BUILD_DIR_NAME="cmake-build-debug"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--debug]"
            echo "  --debug, -d    Build in debug mode"
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

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "🔨 Compiling example_creator in $BUILD_TYPE mode..."
echo "Project root: $PROJECT_ROOT"

# Change to project root
cd "$PROJECT_ROOT"

# Create build directory if it doesn't exist
BUILD_DIR="$PROJECT_ROOT/$BUILD_DIR_NAME"
if [ ! -d "$BUILD_DIR" ]; then
    echo "Creating build directory: $BUILD_DIR"
    mkdir -p "$BUILD_DIR"
fi

# Enter build directory
cd "$BUILD_DIR"

# Configure with CMake if not already configured
if [ ! -f "CMakeCache.txt" ]; then
    echo "⚙️  Configuring project with CMake ($BUILD_TYPE)..."
    cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DPARTH_SOLVER_WITH_TESTS=ON ..
fi

# Build only the example_creator target
echo "🔧 Building example_creator..."
make PARTH_example_creator -j$(nproc)

# Check if compilation was successful
EXECUTABLE="$BUILD_DIR/tests/PARTH_example_creator"
if [ -f "$EXECUTABLE" ]; then
    echo "✅ Compilation successful!"
    echo "Executable: $EXECUTABLE"
else
    echo "❌ Compilation failed - executable not found"
    exit 1
fi
