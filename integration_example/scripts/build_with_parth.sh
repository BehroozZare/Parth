#!/bin/bash

# Build script for Parth Integration Example
# This script configures and builds the integration example using CMake

set -e  # Exit on any error

# Function to find CMake
find_cmake() {
    # Check common CMake locations
    local cmake_paths=(
        "cmake"                                      # In PATH
        "/usr/local/bin/cmake"                      # Homebrew default
        "/opt/homebrew/bin/cmake"                   # Homebrew on Apple Silicon
        "/Applications/CMake.app/Contents/bin/cmake" # CMake.app
    )
    
    for cmake_path in "${cmake_paths[@]}"; do
        if command -v "$cmake_path" >/dev/null 2>&1; then
            echo "$cmake_path"
            return 0
        fi
    done
    
    return 1
}

# Find CMake executable
CMAKE_EXECUTABLE=$(find_cmake)
if [ $? -ne 0 ]; then
    echo "❌ Error: CMake not found!"
    echo ""
    echo "Please install CMake using one of these methods:"
    echo "  1. Homebrew: brew install cmake"
    echo "  2. Download from: https://cmake.org/download/"
    echo "  3. MacPorts: sudo port install cmake"
    echo ""
    exit 1
fi

echo "Using CMake: $CMAKE_EXECUTABLE"

echo "============================================"
echo "Building Parth Integration Example"
echo "============================================"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Script directory: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"

# Create build directory
BUILD_DIR="$PROJECT_ROOT/build"
echo "Build directory: $BUILD_DIR"

if [ -d "$BUILD_DIR" ]; then
    echo "Removing existing build directory..."
    rm -rf "$BUILD_DIR"
fi

echo "Creating build directory..."
mkdir -p "$BUILD_DIR"

# Change to build directory
cd "$BUILD_DIR"

echo "============================================"
echo "Configuring with CMake..."
echo "============================================"

# Configure the project
"$CMAKE_EXECUTABLE" .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17

echo "============================================"
echo "Building the project..."
echo "============================================"

# Build the project
"$CMAKE_EXECUTABLE" --build . --config Release --parallel

echo "============================================"
echo "Build completed successfully!"
echo "============================================"

# Check if executable was created
EXECUTABLE="$BUILD_DIR/use_parth"
if [ -f "$EXECUTABLE" ]; then
    echo "✅ Executable created: $EXECUTABLE"
    echo ""
    echo "To run the example:"
    echo "  cd $BUILD_DIR"
    echo "  ./use_parth"
    echo ""
else
    echo "❌ Error: Executable not found at $EXECUTABLE"
    exit 1
fi

echo "Build script completed successfully!"
