//
//  CHOLMODSolver.cpp
//  IPC
//
//  Created by Minchen Li on 6/22/18.
//

#ifdef PARTH_WITH_ACCELERATE

#include "AccelerateSolver.hpp"
#include "omp.h"


namespace PARTH_SOLVER {
    AccelerateSolver::~AccelerateSolver() {
        if (sym_defined) {
            SparseCleanup(symbolic_info);
        }

        if (factor_defined) {
            SparseCleanup(numeric_info);
        }

    }

    AccelerateSolver::AccelerateSolver() {
        sym_defined = false;
        factor_defined = false;

        this->activate_parth = activate_parth;

        if (activate_parth) {
            std::cout << "Activating Parth" << std::endl;
        } else {
            std::cout << "Deactivating Parth" << std::endl;
        }
        parth.setVerbose(true);
        parth.setNDLevels(6);
        parth.setNumberOfCores(10);
    }

    void AccelerateSolver::setMatrix(int *p, int *i, double *x, int A_N, int NNZ, int *Mp, int *Mi,
                                     int M_N, std::vector<int> &new_to_old_map) {
        assert(p[A_N] == NNZ);
        this->N = A_N;
        this->NNZ = NNZ;
        assert(A_N % M_N == 0);
        this->setSimulationDIM(A_N / M_N);
        if (new_to_old_map.empty()) {
            parth.setMesh(M_N, Mp, Mi);
        } else {
            parth.setMesh(M_N, Mp, Mi, new_to_old_map);
        }

        const Eigen::Index nColumnsStarts = N + 1;

        columnStarts.resize(nColumnsStarts);

        for (Eigen::Index i = 0; i < nColumnsStarts; i++) columnStarts[i] = p[i];

        SparseAttributes_t attributes{};
        attributes.transpose = false;
        attributes.triangle = SparseLowerTriangle;
        attributes.kind = SparseSymmetric;

        SparseMatrixStructure structure{};
        structure.attributes = attributes;
        structure.rowCount = static_cast<int>(N);
        structure.columnCount = static_cast<int>(N);
        structure.blockSize = 1;
        structure.columnStarts = columnStarts.data();
        structure.rowIndices = const_cast<int *>(i);


        A.structure = structure;
        A.data = const_cast<double *>(x);
    }

    void AccelerateSolver::innerAnalyze_pattern(bool no_perm) {
        std::vector<int> perm_inv;
        if (sym_defined) {
            SparseCleanup(symbolic_info);
            sym_defined = false;
        }

        if (activate_parth) {
            perm_inv.clear();
            perm_inv.resize(perm.size());
            for (int i = 0; i < perm.size(); i++) {
                perm_inv[perm[i]] = i;
            }
        }


        opts.control = SparseDefaultControl;
        if (activate_parth) {
            assert(perm.size() == N);
            opts.order = perm_inv.data();
            opts.orderMethod = SparseOrderUser;
        } else {
            if (Base::reorderingType == ReorderingType::METIS) {
                opts.order = nullptr;
                opts.orderMethod = SparseOrderMetis;
                std::cout << "*** ACCELERATE: Choosing METIS" << std::endl;
            } else {
                opts.order = nullptr;
                opts.orderMethod = SparseOrderAMD;
                std::cout << "*** ACCELERATE: Choosing AMD" << std::endl;
            }
        }

        opts.ignoreRowsAndColumns = nullptr;
        opts.malloc = malloc;
        opts.free = free;
        opts.reportError = nullptr;

        symbolic_info = SparseFactor(SparseFactorizationCholesky, A.structure, opts);
        status = symbolic_info.status;
        if (status != SparseStatusOK) {
            std::cerr << "Symbolic factorization returned with error" << std::endl;
        }
        sym_defined = true;
        L_NNZ = symbolic_info.factorSize_Double / 8;
    }

    bool AccelerateSolver::innerFactorize(void) {
        if (factor_defined) {
            SparseCleanup(numeric_info);
            factor_defined = false;
        }

        numeric_info = SparseFactor(symbolic_info, A);
        status = numeric_info.status;
        if (status != SparseStatusOK) {
            std::cerr << "Cholesky factorization returned with error" << std::endl;
            return false;
        }
        factor_defined = true;
        return true;
    }

    void AccelerateSolver::innerSolve(Eigen::VectorXd &rhs, Eigen::VectorXd &result) {
        if (result.rows() != rhs.rows()) {
            result.resize(rhs.rows());
        }
        DenseVector_Double xmat{};

        xmat.count = result.size();
        xmat.data = result.data();

        DenseVector_Double bmat{};
        bmat.count = rhs.size();
        bmat.data = rhs.data();
        SparseSolve(numeric_info, bmat, xmat);
    }

    void AccelerateSolver::innerSolve(Eigen::MatrixXd &rhs, Eigen::MatrixXd &result) {
        if (numeric_info.status != SparseStatusOK) {
            std::cerr << "invalid numeric_info - solve is aborted \n" << std::endl;
            return;
        }

        result.resize(rhs.rows(), rhs.cols());

        double* b_ptr = const_cast<double*>(rhs.derived().data());
        double* x_ptr = const_cast<double*>(result.derived().data());

        DenseMatrix_Double xmat{};
        xmat.columnCount = static_cast<int>(rhs.cols());
        xmat.rowCount = static_cast<int>(rhs.rows());
        xmat.columnStride = xmat.rowCount;
        xmat.data = x_ptr;

        DenseMatrix_Double bmat{};
        bmat.columnCount = static_cast<int>(rhs.cols());
        bmat.rowCount = static_cast<int>(rhs.rows());
        bmat.columnStride = bmat.rowCount;
        bmat.data = b_ptr;

        SparseSolve(numeric_info, bmat, xmat);

        if (numeric_info.status != SparseStatusOK) {
            std::cerr << "Solve failed - solve is aborted \n" << std::endl;
            return;
        }
    }

    void AccelerateSolver::resetSolver() {
        Base::resetParth();
        if (sym_defined) {
            SparseCleanup(symbolic_info);
            sym_defined = false;
        }

        if (factor_defined) {
            SparseCleanup(numeric_info);
            factor_defined = false;
        }
    }

    LinSysSolverType AccelerateSolver::type() const { return LinSysSolverType::ACCELERATE; };
}

#endif
