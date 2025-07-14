
#pragma once

#ifdef PARTH_WITH_SYMPILER

#include "LinSysSolver.hpp"

#include "Parth.h"

#include <sympiler_cholesky.h>
#include "cholmod.h"

#include <Eigen/Eigen>

#include <set>
#include <vector>

namespace PARTH_SOLVER {

class SympilerSolver : public LinSysSolver {
    typedef LinSysSolver Base; // The class
public:                // Access specifier
  sym_lib::parsy::CSC *A;
  sym_lib::parsy::SolverSettings *sym_chol;
  int *Ai, *Ap;
  double *Ax, *bx, *solutionx, *x_cdx, *y_cdx;



    ~SympilerSolver();
    SympilerSolver();

    void setMatrix(int *p, int *i, double *x, int A_N, int NNZ, int *Mp, int *Mi,
                   int M_N, std::vector<int> &new_to_old_map) override;
    void innerAnalyze_pattern(bool no_perm) override;
    bool innerFactorize(void) override;
    void innerSolve(Eigen::VectorXd &rhs, Eigen::VectorXd &result) override;
    void innerSolve(Eigen::MatrixXd &rhs, Eigen::MatrixXd &result) override;
    void resetSolver() override;
    virtual LinSysSolverType type() const override;

};

}

#endif
