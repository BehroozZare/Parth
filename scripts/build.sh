#!/bin/bash

# Parth Simple Build Script
# This script builds the Parth project with minimal user input

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
BUILD_TYPE="Release"
JOBS=$(nproc)

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to get user input with default
get_input() {
    local prompt="$1"
    local default="$2"
    local var_name="$3"
    
    echo -n "$prompt [$default]: "
    read -r input
    
    if [ -z "$input" ]; then
        eval "$var_name=\"$default\""
    else
        eval "$var_name=\"$input\""
    fi
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    if ! command -v cmake &> /dev/null; then
        print_error "CMake is not installed. Please install CMake 3.14 or higher."
        exit 1
    fi
    
    cmake_version=$(cmake --version | head -n1 | cut -d" " -f3)
    print_status "Found CMake version: $cmake_version"
    
    if ! command -v make &> /dev/null; then
        print_error "Make is not installed."
        exit 1
    fi
    
    if ! command -v g++ &> /dev/null && ! command -v clang++ &> /dev/null; then
        print_error "No C++ compiler found. Please install g++ or clang++."
        exit 1
    fi
    
    # Check for optional dependencies
    if ! pkg-config --exists eigen3 &> /dev/null; then
        print_warning "Eigen3 not found via pkg-config. Will try to download automatically."
    else
        eigen_version=$(pkg-config --modversion eigen3)
        print_status "Found Eigen3 version: $eigen_version"
    fi
    
    if ! command -v pkg-config &> /dev/null || ! pkg-config --exists metis &> /dev/null; then
        print_warning "METIS not found. Some ordering features may not work."
    else
        print_status "Found METIS library"
    fi
    
    print_success "Prerequisites check passed"
}

# Function to build examples
build_examples() {
    if [ -d "examples/api_demos/permutation_computation" ]; then
        print_status "Building example executables..."
        
        # Build permutation example
        cd examples/api_demos/permutation_computation
        if [ ! -f "CMakeLists.txt" ]; then
            print_status "Creating CMakeLists.txt for permutation example..."
            cat > CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.14)
project(permutation_example)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the Parth library in the main build
find_package(parth REQUIRED PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake-build-${CMAKE_BUILD_TYPE,,}/lib/cmake/parth NO_DEFAULT_PATH)

# Create the executable
add_executable(permutation_demo permutation.cpp)
target_link_libraries(permutation_demo Parth::parth)

# Copy to main build bin directory
set_target_properties(permutation_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake-build-${CMAKE_BUILD_TYPE,,}/bin
)
EOF
        fi
        
        mkdir -p build
        cd build
        cmake .. -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
        make -j"$JOBS"
        cd ../../..
        
        print_success "Examples built successfully"
    fi
}

# Function to build
build_project() {
    local build_dir="cmake-build-${BUILD_TYPE,,}"
    
    print_status "Building $BUILD_TYPE configuration..."
    
    # Create build directory
    mkdir -p "$build_dir"
    cd "$build_dir"
    
    # Configure CMake with Parth-specific options
    print_status "Configuring CMake..."
    cmake .. \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DPARTH_WITH_METIS=ON \
        -DPARTH_WITH_CHOLMOD=ON \
        -DPARTH_WITH_MKL="${MKL_OPTION:-ON}" \
        -DPARTH_WITH_ACCELERATE=ON \
        -DPARTH_WITH_OPENGL="${OPENGL_OPTION:-ON}" \
        -DPARTH_WITH_PARTH=ON \
        -DPARTH_WITH_DEMO="${DEMO_OPTION:-OFF}" \
        -DPARTH_WITH_SPMP=OFF \
        -DPARTH_WITH_DAGP=OFF \
        -DPARTH_WITH_STRUMPACK=OFF \
        -DPARTH_WITH_SYMPILER=OFF
    
    # Build
    print_status "Building with $JOBS jobs..."
    make -j"$JOBS"
    
    # Create bin directory if it doesn't exist
    mkdir -p bin
    
    print_success "Build completed successfully!"
    
    # Go back to root
    cd ..
    
    # Build examples
    build_examples
}

# Function to install system dependencies
install_dependencies() {
    print_status "Checking if we can install missing dependencies..."
    
    # Check for package manager
    if command -v apt-get &> /dev/null; then
        print_status "Detected apt package manager (Ubuntu/Debian)"
        echo -n "Install missing dependencies automatically? [y/N]: "
        read -r install_deps
        
        if [[ "$install_deps" =~ ^[Yy]$ ]]; then
            sudo apt-get update
            sudo apt-get install -y \
                build-essential \
                cmake \
                libeigen3-dev \
                libsuitesparse-dev \
                libmetis-dev \
                libomp-dev \
                pkg-config
            print_success "Dependencies installed"
        fi
    elif command -v dnf &> /dev/null; then
        print_status "Detected dnf package manager (Fedora/RHEL)"
        echo -n "Install missing dependencies automatically? [y/N]: "
        read -r install_deps
        
        if [[ "$install_deps" =~ ^[Yy]$ ]]; then
            sudo dnf install -y \
                gcc-c++ \
                cmake \
                eigen3-devel \
                suitesparse-devel \
                METIS-devel \
                libomp-devel \
                pkgconfig
            print_success "Dependencies installed"
        fi
    elif command -v pacman &> /dev/null; then
        print_status "Detected pacman package manager (Arch Linux)"
        echo -n "Install missing dependencies automatically? [y/N]: "
        read -r install_deps
        
        if [[ "$install_deps" =~ ^[Yy]$ ]]; then
            sudo pacman -S --needed \
                base-devel \
                cmake \
                eigen \
                suitesparse \
                metis \
                openmp \
                pkgconf
            print_success "Dependencies installed"
        fi
    else
        print_warning "No supported package manager found. Please install dependencies manually:"
        echo "  - CMake 3.14+"
        echo "  - C++17 compiler (g++ or clang++)"
        echo "  - Eigen3"
        echo "  - SuiteSparse (optional)"
        echo "  - METIS (optional)"
        echo "  - OpenMP (optional)"
    fi
}

# Main execution
main() {
    echo "Parth Build Script"
    echo "=================="
    echo
    
    # Get the script directory and project root
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
    
    # Change to project root directory
    cd "$PROJECT_ROOT"
    
    # Check if we're in the right directory
    if [ ! -f "CMakeLists.txt" ] || ! grep -q "PARTH" CMakeLists.txt; then
        print_error "Could not find Parth project root directory"
        print_error "Expected to find CMakeLists.txt with PARTH_SOLVER in: $PROJECT_ROOT"
        exit 1
    fi
    
    print_status "Running from project root: $PROJECT_ROOT"
    
    # Check prerequisites
    if ! check_prerequisites; then
        echo
        install_dependencies
        echo
        check_prerequisites
    fi
    
    # Get build type
    echo "Build type options:"
    echo "  Release - Optimized build (default)"
    echo "  Debug   - Debug build with symbols"
    echo "  RelWithDebInfo - Release with debug info"
    echo
    get_input "Select build type" "$BUILD_TYPE" "BUILD_TYPE"
    
    # Advanced options
    echo
    echo -n "Configure advanced options? [y/N]: "
    read -r advanced
    
    if [[ "$advanced" =~ ^[Yy]$ ]]; then
        echo
        echo "Advanced Build Options:"
        echo "Note: These require specific libraries to be installed"
        echo
        
        echo -n "Enable Intel MKL support? [Y/n]: "
        read -r mkl_option
        MKL_OPTION="ON"
        if [[ "$mkl_option" =~ ^[Nn]$ ]]; then
            MKL_OPTION="OFF"
        fi
        
        echo -n "Enable OpenGL visualization? [Y/n]: "
        read -r opengl_option
        OPENGL_OPTION="ON"
        if [[ "$opengl_option" =~ ^[Nn]$ ]]; then
            OPENGL_OPTION="OFF"
        fi
        
        echo -n "Enable paper usage examples? [y/N]: "
        read -r demo_option
        DEMO_OPTION="OFF"
        if [[ "$demo_option" =~ ^[Yy]$ ]]; then
            DEMO_OPTION="ON"
        fi
    fi
    
    # Confirm build
    echo
    echo "Build Configuration:"
    echo "  Build Type: $BUILD_TYPE"
    echo "  Jobs: $JOBS"
    echo "  METIS: ON"
    echo "  CHOLMOD: ON"
    echo "  MKL: ${MKL_OPTION:-ON}"
    echo "  OpenGL: ${OPENGL_OPTION:-ON}"
    echo "  Demo/Paper Examples: ${DEMO_OPTION:-OFF}"
    echo
    
    echo -n "Proceed with build? [Y/n]: "
    read -r confirm
    
    if [[ "$confirm" =~ ^[Nn]$ ]]; then
        print_status "Build cancelled by user"
        exit 0
    fi
    
    # Build the project
    echo
    build_project
    
    # Show results
    echo
    print_success "Build completed successfully!"
    echo
    echo "Build Summary:"
    echo "=============="
    echo "✓ $BUILD_TYPE build: cmake-build-${BUILD_TYPE,,}/"
    echo "✓ Core Parth library: cmake-build-${BUILD_TYPE,,}/libparth.a"
    echo "✓ Full solver library: cmake-build-${BUILD_TYPE,,}/libPARTH_SOLVER_lib.a"
    
    # List built executables
    if [ -d "cmake-build-${BUILD_TYPE,,}/bin" ] && [ "$(ls -A cmake-build-${BUILD_TYPE,,}/bin 2>/dev/null)" ]; then
        echo "✓ Example executables:"
        for exe in cmake-build-${BUILD_TYPE,,}/bin/*; do
            if [ -x "$exe" ]; then
                echo "  - $(basename "$exe")"
            fi
        done
    fi
    
    if [ "${DEMO_OPTION:-OFF}" = "ON" ] && [ -f "cmake-build-${BUILD_TYPE,,}/PARTH_SOLVER_IPCBenchmark" ]; then
        echo "✓ Paper/Demo executables:"
        for exe in cmake-build-${BUILD_TYPE,,}/PARTH_SOLVER_*; do
            if [ -x "$exe" ]; then
                echo "  - $(basename "$exe")"
            fi
        done
    fi
    
    echo
    echo "Usage:"
    echo "======"
    echo "To run the permutation example:"
    echo "  ./cmake-build-${BUILD_TYPE,,}/bin/permutation_demo"
    echo
    echo "To run the build script again:"
    echo "  ./scripts/build.sh"
    echo
    echo "To use Parth in your own project:"
    echo "  find_package(parth REQUIRED PATHS ./cmake-build-${BUILD_TYPE,,}/lib/cmake/parth)"
    echo "  target_link_libraries(your_target Parth::parth)"
    echo
    echo "To clean: rm -rf cmake-build-*"
    echo
    print_success "Ready to use Parth!"
}

# Run main function
main "$@"
