
#pragma once

#ifdef PARTH_WITH_ACCELERATE

#include "LinSysSolver.hpp"

#include <Accelerate/Accelerate.h>
#include "cholmod.h"

#include <Eigen/Eigen>

#include <set>
#include <vector>

namespace PARTH_SOLVER {

class AccelerateSolver : public LinSysSolver {
    typedef LinSysSolver Base; // The class
public:                // Access specifier
    bool sym_defined;
    bool factor_defined;
    //Accelerate Stuff
    std::vector<long> columnStarts;
    SparseMatrix_Double A;

    SparseOpaqueSymbolicFactorization symbolic_info;
    SparseOpaqueFactorization_Double numeric_info;
    SparseSymbolicFactorOptions opts{};
    SparseStatus_t status;


    ~AccelerateSolver();
    AccelerateSolver();

    void setMatrix(int *p, int *i, double *x, int A_N, int NNZ, int *Mp, int *Mi,
                   int M_N, std::vector<int> &new_to_old_map) override;
    void innerAnalyze_pattern(bool no_perm) override;
    bool innerFactorize(void) override;
    void innerSolve(Eigen::VectorXd &rhs, Eigen::VectorXd &result) override;
    void innerSolve(Eigen::MatrixXd& rhs, Eigen::MatrixXd& result) override;
    void resetSolver() override;
    virtual LinSysSolverType type() const override;

};

}

#endif
