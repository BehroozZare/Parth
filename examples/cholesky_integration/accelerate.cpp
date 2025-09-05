//
// Accelerate Framework Cholesky Integration Demo - Shows how to use Parth with Apple's Accelerate
//
// This example demonstrates:
// 1. Setting up a linear system with Parth permutation
// 2. Using Accelerate LAPACK with Parth ordering
// 3. Comparing performance with different orderings
//

#ifdef PARTH_WITH_ACCELERATE

#include <parth/parth.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <Accelerate/Accelerate.h>

class AccelerateCholeskyDemo {
private:
    std::vector<int> Mp, Mi;
    std::vector<double> Mx;
    int n;
    
    void createTestMatrix() {
        // Create a simple 2D Laplacian matrix (5-point stencil)
        n = 50; // 7x7 grid (smaller for dense Cholesky)
        Mp.resize(n + 1);
        
        // Create dense matrix from sparse structure first
        std::vector<std::vector<double>> dense(n, std::vector<double>(n, 0.0));
        
        // Build 2D grid adjacency
        int grid_size = static_cast<int>(std::sqrt(n));
        for (int i = 0; i < grid_size; i++) {
            for (int j = 0; j < grid_size; j++) {
                if (i * grid_size + j >= n) break;
                
                int node = i * grid_size + j;
                dense[node][node] = 4.0;
                
                // Add neighbors
                if (i > 0 && (i-1)*grid_size + j < n) 
                    dense[node][(i-1)*grid_size + j] = -1.0;
                if (i < grid_size-1 && (i+1)*grid_size + j < n) 
                    dense[node][(i+1)*grid_size + j] = -1.0;
                if (j > 0) 
                    dense[node][i*grid_size + (j-1)] = -1.0;
                if (j < grid_size-1 && i*grid_size + (j+1) < n) 
                    dense[node][i*grid_size + (j+1)] = -1.0;
            }
        }
        
        // Convert to sparse CSR format for Parth
        Mp[0] = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (std::abs(dense[i][j]) > 1e-12) {
                    Mi.push_back(j);
                    Mx.push_back(dense[i][j]);
                }
            }
            Mp[i + 1] = Mi.size();
        }
        
        std::cout << "Created " << n << "x" << n << " matrix with " 
                  << Mi.size() << " non-zeros" << std::endl;
    }
    
    void solveDenseCholesky(const std::vector<int>& perm, const std::string& description) {
        std::cout << "\n" << description << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Convert sparse to dense format
        std::vector<double> A(n * n, 0.0);
        for (int i = 0; i < n; i++) {
            for (int k = Mp[i]; k < Mp[i + 1]; k++) {
                int j = Mi[k];
                A[i * n + j] = Mx[k];
            }
        }
        
        // Apply permutation if provided
        if (!perm.empty()) {
            std::vector<double> A_perm(n * n);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    A_perm[perm[i] * n + perm[j]] = A[i * n + j];
                }
            }
            A = A_perm;
        }
        
        // Create RHS
        std::vector<double> b(n, 1.0), x(n);
        
        // Apply permutation to RHS
        if (!perm.empty()) {
            std::vector<double> b_perm(n);
            for (int i = 0; i < n; i++) {
                b_perm[perm[i]] = b[i];
            }
            b = b_perm;
        }
        x = b; // Copy for solving
        
        // LAPACK Cholesky factorization (DPOTRF)
        char uplo = 'L';
        __CLPK_integer info;
        __CLPK_integer lapack_n = n;
        
        dpotrf_(&uplo, &lapack_n, A.data(), &lapack_n, &info);
        
        if (info != 0) {
            std::cout << "  Cholesky factorization failed: " << info << std::endl;
            return;
        }
        
        // Solve using factorization (DPOTRS)
        __CLPK_integer nrhs = 1;
        dpotrs_(&uplo, &lapack_n, &nrhs, A.data(), &lapack_n, x.data(), &lapack_n, &info);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        if (info != 0) {
            std::cout << "  Solve failed: " << info << std::endl;
            return;
        }
        
        // Permute solution back
        if (!perm.empty()) {
            std::vector<double> x_orig(n);
            for (int i = 0; i < n; i++) {
                x_orig[i] = x[perm[i]];
            }
            x = x_orig;
        }
        
        std::cout << "  Solve time: " << duration.count() << " μs" << std::endl;
        std::cout << "  Solution norm: " << cblas_dnrm2(n, x.data(), 1) << std::endl;
    }
    
public:
    void run() {
        std::cout << "=== Accelerate Framework Cholesky Demo ===" << std::endl;
        
        // Create test matrix
        createTestMatrix();
        
        // Solve without Parth ordering
        std::vector<int> empty_perm;
        solveDenseCholesky(empty_perm, "Accelerate LAPACK with natural ordering:");
        
        // Setup Parth
        PARTH::Parth parth;
        parth.setReorderingType(PARTH::ReorderingType::METIS);
        parth.setVerbose(false);
        parth.setNDLevels(3);
        
        // Set mesh data (matrix structure)
        parth.setMeshPointers(n, Mp.data(), Mi.data());
        
        // Compute Parth permutation
        std::cout << "\nComputing Parth permutation..." << std::endl;
        std::vector<int> perm;
        parth.computePermutation(perm, 1);
        
        // Solve with Parth ordering
        solveDenseCholesky(perm, "Accelerate LAPACK with Parth ordering:");
        
        // Show timing information
        std::cout << "\nParth timing information:" << std::endl;
        parth.printTiming();
        
        std::cout << "\n=== Demo completed! ===" << std::endl;
    }
};

int main() {
    AccelerateCholeskyDemo demo;
    demo.run();
    return 0;
}

#else
#include <iostream>
int main() {
    std::cout << "This demo requires Accelerate Framework support. Available only on macOS." << std::endl;
    return 1;
}
#endif
