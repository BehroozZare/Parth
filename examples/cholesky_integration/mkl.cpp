//
// MKL Cholesky Integration Demo - Shows how to use Parth with Intel MKL
//
// This example demonstrates:
// 1. Setting up a linear system with Parth permutation
// 2. Using MKL PARDISO solver with Parth ordering
// 3. Comparing performance with and without Parth
//

#ifdef PARTH_WITH_MKL

#include <parth/parth.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <mkl.h>
#include <mkl_pardiso.h>

class MKLCholeskyDemo {
private:
    std::vector<int> Mp, Mi;
    std::vector<double> Mx;
    int n;
    
    void createTestMatrix() {
        // Create a simple 2D Laplacian matrix (5-point stencil)
        n = 100; // 10x10 grid
        Mp.resize(n + 1);
        
        std::vector<std::vector<std::pair<int, double>>> adj(n);
        
        // Build 2D grid adjacency
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                int node = i * 10 + j;
                double diag_val = 4.0;
                
                // Add neighbors
                if (i > 0) adj[node].push_back({(i-1)*10 + j, -1.0});
                if (i < 9) adj[node].push_back({(i+1)*10 + j, -1.0});
                if (j > 0) adj[node].push_back({i*10 + (j-1), -1.0});
                if (j < 9) adj[node].push_back({i*10 + (j+1), -1.0});
                
                // Add diagonal
                adj[node].push_back({node, diag_val});
            }
        }
        
        // Convert to CSR
        Mp[0] = 0;
        for (int i = 0; i < n; i++) {
            // Sort by column index
            std::sort(adj[i].begin(), adj[i].end());
            Mp[i + 1] = Mp[i] + adj[i].size();
            for (auto& entry : adj[i]) {
                Mi.push_back(entry.first);
                Mx.push_back(entry.second);
            }
        }
        
        std::cout << "Created " << n << "x" << n << " Laplacian matrix with " 
                  << Mi.size() << " non-zeros" << std::endl;
    }
    
    void solveMKL(const std::vector<int>& perm, const std::string& description) {
        std::cout << "\n" << description << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // MKL PARDISO parameters
        void* pt[64] = {0};
        MKL_INT iparm[64] = {0};
        MKL_INT maxfct = 1, mnum = 1, phase = 13, msglvl = 0;
        MKL_INT mtype = 2; // Real symmetric positive definite
        MKL_INT nrhs = 1, error = 0;
        
        // Set default parameters
        iparm[0] = 1;  // Use non-default values
        iparm[1] = 2;  // Fill-in reordering from METIS
        iparm[3] = 0;  // No iterative-direct algorithm
        iparm[4] = 0;  // No user permutation
        iparm[7] = 2;  // Max numbers of iterative refinement steps
        iparm[9] = 13; // Perturb pivot threshold
        iparm[10] = 1; // Use nonsymmetric permutation
        iparm[17] = -1; // Report number of non-zeros in factors
        iparm[18] = -1; // Report Mflops for LU factorization
        
        // Apply permutation if provided
        std::vector<MKL_INT> mkl_perm;
        if (!perm.empty()) {
            mkl_perm.resize(perm.size());
            for (size_t i = 0; i < perm.size(); i++) {
                mkl_perm[i] = perm[i] + 1; // MKL uses 1-based indexing
            }
            iparm[4] = 1; // Use user permutation
        }
        
        // Convert to MKL types
        std::vector<MKL_INT> mkl_Mp(Mp.begin(), Mp.end());
        std::vector<MKL_INT> mkl_Mi(Mi.begin(), Mi.end());
        
        // Convert to 1-based indexing
        for (auto& val : mkl_Mp) val++;
        for (auto& val : mkl_Mi) val++;
        
        MKL_INT mkl_n = n;
        
        // Create RHS
        std::vector<double> b(n, 1.0), x(n, 0.0);
        
        // Call PARDISO
        pardiso(pt, &maxfct, &mnum, &mtype, &phase, &mkl_n,
                Mx.data(), mkl_Mp.data(), mkl_Mi.data(),
                mkl_perm.empty() ? nullptr : mkl_perm.data(),
                &nrhs, iparm, &msglvl, b.data(), x.data(), &error);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        if (error != 0) {
            std::cout << "PARDISO error: " << error << std::endl;
            return;
        }
        
        std::cout << "  Solve time: " << duration.count() << " ms" << std::endl;
        std::cout << "  Fill-in: " << iparm[17] << " non-zeros" << std::endl;
        std::cout << "  Peak memory: " << std::max(iparm[14], iparm[15] + iparm[16]) << " KB" << std::endl;
        
        // Cleanup
        phase = -1;
        pardiso(pt, &maxfct, &mnum, &mtype, &phase, &mkl_n,
                nullptr, nullptr, nullptr, nullptr, &nrhs, iparm, &msglvl,
                nullptr, nullptr, &error);
    }
    
public:
    void run() {
        std::cout << "=== MKL Cholesky Integration Demo ===" << std::endl;
        
        // Create test matrix
        createTestMatrix();
        
        // Solve without Parth ordering
        std::vector<int> empty_perm;
        solveMKL(empty_perm, "MKL PARDISO with default ordering:");
        
        // Setup Parth
        PARTH::Parth parth;
        parth.setReorderingType(PARTH::ReorderingType::METIS);
        parth.setVerbose(false);
        parth.setNDLevels(4);
        
        // Set mesh data (matrix structure)
        parth.setMeshPointers(n, Mp.data(), Mi.data());
        
        // Compute Parth permutation
        std::cout << "\nComputing Parth permutation..." << std::endl;
        std::vector<int> perm;
        parth.computePermutation(perm, 1);
        
        // Solve with Parth ordering
        solveMKL(perm, "MKL PARDISO with Parth ordering:");
        
        std::cout << "\n=== Demo completed! ===" << std::endl;
    }
};

int main() {
    MKLCholeskyDemo demo;
    demo.run();
    return 0;
}

#else
#include <iostream>
int main() {
    std::cout << "This demo requires MKL support. Please build with -DPARTH_SOLVER_WITH_MKL=ON" << std::endl;
    return 1;
}
#endif
