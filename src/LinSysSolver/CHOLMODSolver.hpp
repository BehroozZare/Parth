
#pragma once

#ifdef PARTH_WITH_CHOLMOD

#include "LinSysSolver.hpp"

#include <cholmod.h>
#include "cholmod.h"

#include <Eigen/Eigen>

#include <set>
#include <vector>

namespace PARTH_SOLVER {

class CHOLMODSolver : public LinSysSolver {
    typedef LinSysSolver Base; // The class
public:                // Access specifier
    cholmod_common cm;
    cholmod_sparse *A;
    cholmod_factor *L;
    cholmod_dense *b;

    cholmod_dense *x_solve;

    void *Ai, *Ap, *Ax, *bx;


    ~CHOLMODSolver();
    CHOLMODSolver();

    void setMatrix(int *p, int *i, double *x, int A_N, int NNZ, int *Mp, int *Mi,
                   int M_N, std::vector<int> &new_to_old_map) override;
    void innerAnalyze_pattern(bool no_perm) override;
    bool innerFactorize(void) override;
    void innerSolve(Eigen::VectorXd &rhs, Eigen::VectorXd &result) override;
    void innerSolve(Eigen::MatrixXd &rhs, Eigen::MatrixXd &result) override;
    void resetSolver() override;
    virtual LinSysSolverType type() const override;

    void cholmod_clean_memory();

};

}

#endif
