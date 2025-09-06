#include <iostream>
#include <string>
#include <parth/parth.h>
#include <CLI/CLI.hpp>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

/*
 * PARTH Quick Start Example: Demonstrating Computational Reuse
 * 
 * This example demonstrates the key capability of PARTH: computational reuse when
 * processing matrices with similar sparsity patterns but different non-zero entries.
 * 
 * Workflow:
 * 1. Load an original matrix and compute its permutation from scratch
 * 2. Load a modified matrix (with additional non-zero entries) 
 * 3. Compute permutation for the modified matrix while reusing previous computations
 * 
 * The second computation should be significantly faster due to PARTH's ability to
 * reuse intermediate results from the first matrix when the sparsity pattern changes
 * minimally (e.g., only a few new edges are added to the graph representation).
 */

int main(int argc, char *argv[]) {
    
    // =======================================================================
    // STEP 1: Load the Original Matrix
    // =======================================================================
    // This is our baseline matrix - PARTH will compute the permutation from scratch
    std::cout << "=== STEP 1: Loading Original Matrix ===" << std::endl;
    std::cout << "Loading original matrix..." << std::endl;
    Eigen::SparseMatrix<double> original_matrix;
    std::string original_matrix_path = "/Users/behrooz/Desktop/LastProject/Parth/tests/matrices/original_matrix.mtx";
    if (!Eigen::loadMarket(original_matrix, original_matrix_path)) {
        std::cerr << "Failed to load original matrix from: " << original_matrix_path << std::endl;
        return 1;
    }
    std::cout << "Original matrix loaded successfully. Size: " << original_matrix.rows() << "x" << original_matrix.cols() 
              << ", Non-zeros: " << original_matrix.nonZeros() << std::endl;

    // =======================================================================
    // STEP 2: Load the Modified Matrix
    // =======================================================================
    // This matrix has additional non-zero entries compared to the original matrix.
    // It represents the same underlying problem but with a slightly modified sparsity pattern
    // (e.g., new connections/edges added to the graph structure).
    std::cout << "\n=== STEP 2: Loading Modified Matrix ===" << std::endl;
    std::cout << "Loading modified matrix (with additional entries)..." << std::endl;
    Eigen::SparseMatrix<double> matrix_1;
    std::string matrix_1_path = "/Users/behrooz/Desktop/LastProject/Parth/tests/matrices/2_matrix.mtx";
    if (!Eigen::loadMarket(matrix_1, matrix_1_path)) {
        std::cerr << "Failed to load modified matrix from: " << matrix_1_path << std::endl;
        return 1;
    }
    std::cout << "Modified matrix loaded successfully. Size: " << matrix_1.rows() << "x" << matrix_1.cols() 
              << ", Non-zeros: " << matrix_1.nonZeros() << std::endl;
    
    // Show the difference in sparsity
    int added_entries = matrix_1.nonZeros() - original_matrix.nonZeros();
    std::cout << "Additional non-zero entries: " << added_entries << " (+" 
              << (100.0 * added_entries / original_matrix.nonZeros()) << "%)" << std::endl;

    // =======================================================================
    // STEP 3: First PARTH Computation (From Scratch)
    // =======================================================================
    // Initialize PARTH and compute permutation for the original matrix.
    // This computation starts from scratch - no previous data to reuse.
    std::cout << "\n=== STEP 3: Computing Permutation for Original Matrix ===" << std::endl;
    std::cout << "Initializing PARTH API..." << std::endl;
    
    PARTH::ParthAPI parth;
    
    // Set the matrix data: provide matrix dimensions and CSC format pointers
    // - rows: number of rows/columns (square matrix)
    // - outerIndexPtr: column pointers in CSC format
    // - innerIndexPtr: row indices in CSC format  
    // - 1: dimension parameter (1 DOF per graph node)
    std::cout << "Setting original matrix data in PARTH..." << std::endl;
    parth.setMatrix(original_matrix.rows(),
                    const_cast<int*>(original_matrix.outerIndexPtr()), 
                    const_cast<int*>(original_matrix.innerIndexPtr()), 1);
    
    // Compute the fill-reducing permutation
    // This will build the hierarchical mesh decomposition (HMD) from scratch
    std::cout << "Computing permutation (from scratch)..." << std::endl;
    std::vector<int> perm;
    parth.computePermutation(perm);
    
    // Display timing information - this shows the "cold start" performance
    std::cout << "=== TIMING FOR ORIGINAL MATRIX (Cold Start) ===" << std::endl;
    parth.printTiming();
    std::cout << "Original matrix permutation computed successfully." << std::endl;
    std::cout << "Permutation vector size: " << perm.size() << std::endl;

    // =======================================================================
    // STEP 4: Second PARTH Computation (With Computational Reuse)
    // =======================================================================
    // Now we process the modified matrix. PARTH will detect similarities with the
    // previous matrix and reuse as much computation as possible.
    // 
    // Key insight: When the sparsity pattern changes minimally (only a few edges added),
    // PARTH can reuse most of the hierarchical decomposition computed in Step 3.
    // This leads to dramatic speedups compared to computing from scratch.
    std::cout << "\n=== STEP 4: Computing Permutation for Modified Matrix ===" << std::endl;
    std::cout << "Setting modified matrix data in PARTH..." << std::endl;
    std::cout << "Note: PARTH will automatically detect changes and reuse previous computations!" << std::endl;
    
    // Update the matrix data - PARTH will internally compare with previous matrix
    // and identify which parts of the hierarchical decomposition can be reused
    parth.setMatrix(matrix_1.rows(),
                    const_cast<int*>(matrix_1.outerIndexPtr()), 
                    const_cast<int*>(matrix_1.innerIndexPtr()), 1);

    // Compute permutation for the modified matrix
    // This computation should be much faster due to reuse of previous results
    std::cout << "Computing permutation (with reuse from previous computation)..." << std::endl;
    parth.computePermutation(perm);
    
    // Display timing information - compare with Step 3 to see the speedup!
    std::cout << "=== TIMING FOR MODIFIED MATRIX (With Reuse) ===" << std::endl;
    parth.printTiming();
    std::cout << "Modified matrix permutation computed successfully." << std::endl;
    std::cout << "Permutation vector size: " << perm.size() << std::endl;

    std::cout << "\nBoth matrices processed successfully!" << std::endl;

    return 0;
}