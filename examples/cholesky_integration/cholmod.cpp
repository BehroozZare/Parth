//
// CHOLMOD Integration Demo - Shows how to use Parth with SuiteSparse CHOLMOD
//
// This example demonstrates:
// 1. Setting up a linear system with Parth permutation
// 2. Using CHOLMOD solver with Parth ordering
// 3. Comparing fill-in and performance
//

#ifdef PARTH_WITH_CHOLMOD

#include <parth/parth.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cholmod.h>

class CHOLMODDemo {
private:
    std::vector<int> Mp, Mi;
    std::vector<double> Mx;
    int n;
    cholmod_common c;
    
    void createTestMatrix() {
        // Create a larger 2D Laplacian matrix for better demonstration
        int grid_size = 20; // 20x20 grid
        n = grid_size * grid_size;
        Mp.resize(n + 1);
        
        std::vector<std::vector<std::pair<int, double>>> adj(n);
        
        // Build 2D grid adjacency
        for (int i = 0; i < grid_size; i++) {
            for (int j = 0; j < grid_size; j++) {
                int node = i * grid_size + j;
                
                // Add neighbors with -1 and diagonal with 4
                if (i > 0) adj[node].push_back({(i-1)*grid_size + j, -1.0});
                if (i < grid_size-1) adj[node].push_back({(i+1)*grid_size + j, -1.0});
                if (j > 0) adj[node].push_back({i*grid_size + (j-1), -1.0});
                if (j < grid_size-1) adj[node].push_back({i*grid_size + (j+1), -1.0});
                
                // Diagonal entry
                double diag_val = 4.0;
                if (i == 0 || i == grid_size-1) diag_val += 1.0; // Boundary conditions
                if (j == 0 || j == grid_size-1) diag_val += 1.0;
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
    
    cholmod_sparse* createCHOLMODMatrix(const std::vector<int>& perm) {
        // Convert CSR to triplet format first
        std::vector<int> Ti, Tj;
        std::vector<double> Tx;
        
        for (int i = 0; i < n; i++) {
            for (int k = Mp[i]; k < Mp[i + 1]; k++) {
                int j = Mi[k];
                // Only store lower triangular part for symmetric matrix
                if (j <= i) {
                    Ti.push_back(i);
                    Tj.push_back(j);
                    Tx.push_back(Mx[k]);
                }
            }
        }
        
        // Apply permutation if provided
        if (!perm.empty()) {
            for (size_t k = 0; k < Ti.size(); k++) {
                Ti[k] = perm[Ti[k]];
                Tj[k] = perm[Tj[k]];
                // Ensure lower triangular
                if (Ti[k] < Tj[k]) std::swap(Ti[k], Tj[k]);
            }
        }
        
        // Create CHOLMOD triplet matrix
        cholmod_triplet *T = cholmod_allocate_triplet(n, n, Ti.size(), 0, CHOLMOD_REAL, &c);
        if (!T) return nullptr;
        
        int *Ti_ptr = static_cast<int*>(T->i);
        int *Tj_ptr = static_cast<int*>(T->j);
        double *Tx_ptr = static_cast<double*>(T->x);
        
        for (size_t k = 0; k < Ti.size(); k++) {
            Ti_ptr[k] = Ti[k];
            Tj_ptr[k] = Tj[k];
            Tx_ptr[k] = Tx[k];
        }
        T->nnz = Ti.size();
        
        // Convert to sparse matrix
        cholmod_sparse *A = cholmod_triplet_to_sparse(T, Ti.size(), &c);
        cholmod_free_triplet(&T, &c);
        
        return A;
    }
    
    void solveCHOLMOD(const std::vector<int>& perm, const std::string& description) {
        std::cout << "\n" << description << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Create CHOLMOD matrix
        cholmod_sparse *A = createCHOLMODMatrix(perm);
        if (!A) {
            std::cout << "  Failed to create CHOLMOD matrix" << std::endl;
            return;
        }
        
        // Create RHS
        cholmod_dense *b = cholmod_allocate_dense(n, 1, n, CHOLMOD_REAL, &c);
        double *b_ptr = static_cast<double*>(b->x);
        for (int i = 0; i < n; i++) b_ptr[i] = 1.0;
        
        // Apply permutation to RHS
        if (!perm.empty()) {
            std::vector<double> b_temp(n);
            for (int i = 0; i < n; i++) b_temp[perm[i]] = b_ptr[i];
            for (int i = 0; i < n; i++) b_ptr[i] = b_temp[i];
        }
        
        // Analyze and factorize
        cholmod_factor *L = cholmod_analyze(A, &c);
        cholmod_factorize(A, L, &c);
        
        auto factor_end = std::chrono::high_resolution_clock::now();
        
        // Solve
        cholmod_dense *x = cholmod_solve(CHOLMOD_A, L, b, &c);
        
        auto end = std::chrono::high_resolution_clock::now();
        
        auto factor_time = std::chrono::duration_cast<std::chrono::milliseconds>(factor_end - start);
        auto solve_time = std::chrono::duration_cast<std::chrono::microseconds>(end - factor_end);
        
        if (c.status != CHOLMOD_OK) {
            std::cout << "  CHOLMOD error occurred" << std::endl;
        } else {
            std::cout << "  Factorization time: " << factor_time.count() << " ms" << std::endl;
            std::cout << "  Solve time: " << solve_time.count() << " μs" << std::endl;
            std::cout << "  Fill-in ratio: " << static_cast<double>(L->nzmax) / A->nzmax << std::endl;
            std::cout << "  FLOP count: " << c.fl << std::endl;
        }
        
        // Cleanup
        cholmod_free_dense(&x, &c);
        cholmod_free_dense(&b, &c);
        cholmod_free_factor(&L, &c);
        cholmod_free_sparse(&A, &c);
    }
    
public:
    CHOLMODDemo() {
        cholmod_start(&c);
        c.print = 0; // Reduce output
    }
    
    ~CHOLMODDemo() {
        cholmod_finish(&c);
    }
    
    void run() {
        std::cout << "=== CHOLMOD Integration Demo ===" << std::endl;
        
        // Create test matrix
        createTestMatrix();
        
        // Solve without Parth ordering
        std::vector<int> empty_perm;
        solveCHOLMOD(empty_perm, "CHOLMOD with AMD ordering:");
        
        // Setup Parth
        PARTH::Parth parth;
        parth.setReorderingType(PARTH::ReorderingType::METIS);
        parth.setVerbose(false);
        parth.setNDLevels(5);
        
        // Set mesh data (matrix structure)
        parth.setMeshPointers(n, Mp.data(), Mi.data());
        
        // Compute Parth permutation
        std::cout << "\nComputing Parth permutation..." << std::endl;
        std::vector<int> perm;
        parth.computePermutation(perm, 1);
        
        // Solve with Parth ordering
        solveCHOLMOD(perm, "CHOLMOD with Parth ordering:");
        
        // Show Parth timing
        std::cout << "\nParth timing information:" << std::endl;
        parth.printTiming();
        
        std::cout << "\n=== Demo completed! ===" << std::endl;
    }
};

int main() {
    CHOLMODDemo demo;
    demo.run();
    return 0;
}

#else
#include <iostream>
int main() {
    std::cout << "This demo requires CHOLMOD support. Please build with -DPARTH_SOLVER_WITH_CHOLMOD=ON" << std::endl;
    return 1;
}
#endif
