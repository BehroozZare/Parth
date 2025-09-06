#include <iostream>
#include <string>
#include <parth/parth.h>
#include <CLI/CLI.hpp>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>


int main(int argc, char *argv[]) {
    // Read the original matrix
    std::cout << "Loading original matrix..." << std::endl;
    Eigen::SparseMatrix<double> original_matrix;
    std::string original_matrix_path = "/Users/behrooz/Desktop/LastProject/Parth/tests/matrices/original_matrix.mtx";
    if (!Eigen::loadMarket(original_matrix, original_matrix_path)) {
        std::cerr << "Failed to load original matrix from: " << original_matrix_path << std::endl;
        return 1;
    }
    std::cout << "Original matrix loaded successfully. Size: " << original_matrix.rows() << "x" << original_matrix.cols() 
              << ", Non-zeros: " << original_matrix.nonZeros() << std::endl;

    // Read the 1_matrix.mtx
    std::cout << "Loading 1_matrix.mtx..." << std::endl;
    Eigen::SparseMatrix<double> matrix_1;
    std::string matrix_1_path = "/Users/behrooz/Desktop/LastProject/Parth/tests/matrices/2_matrix.mtx";
    if (!Eigen::loadMarket(matrix_1, matrix_1_path)) {
        std::cerr << "Failed to load 1_matrix from: " << matrix_1_path << std::endl;
        return 1;
    }
    std::cout << "1_matrix loaded successfully. Size: " << matrix_1.rows() << "x" << matrix_1.cols() 
              << ", Non-zeros: " << matrix_1.nonZeros() << std::endl;

    // Apply Parth for computing permutation on original matrix
    std::cout << "\n=== Computing permutation for original matrix ===" << std::endl;
    PARTH::ParthAPI parth;
    parth.setMatrix(original_matrix.rows(),
                            const_cast<int*>(original_matrix.outerIndexPtr()), 
                            const_cast<int*>(original_matrix.innerIndexPtr()), 1);
    
    std::vector<int> perm;
    parth.computePermutation(perm);
    parth.printTiming();
    std::cout << "Original matrix permutation computed." << std::endl;

    // Apply Parth for computing permutation on 1_matrix
    std::cout << "\n=== Computing permutation for 1_matrix ===" << std::endl;
    parth.setMatrix(matrix_1.rows(),
                     const_cast<int*>(matrix_1.outerIndexPtr()), 
                     const_cast<int*>(matrix_1.innerIndexPtr()), 1);

    parth.computePermutation(perm);
    parth.printTiming();
    std::cout << "1_matrix permutation computed." << std::endl;

    std::cout << "\nBoth matrices processed successfully!" << std::endl;

    return 0;
}