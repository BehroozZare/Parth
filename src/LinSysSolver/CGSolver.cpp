

#include "CGSolver.hpp"

namespace PARTH_SOLVER {
CGSolver::~CGSolver() {

}

CGSolver::CGSolver() {

  parth.clearParth();
  parth.setVerbose(true);
  parth.setNDLevels(6);
  parth.setNumberOfCores(10);
  Base::initVariables();
}


void CGSolver::setMatrix(int *p, int *i, double *x, int A_N, int NNZ, int *Mp,
                          int *Mi, int M_N, std::vector<int> &new_to_old_map) {
        this->N = A_N;
        this->NNZ = NNZ;
        Eigen::Map<Eigen::SparseMatrix<double> > A_lower(A_N,A_N,NNZ,p, // read-write
                                                        i,x);
        coefMtr = A_lower;
}

void CGSolver::innerAnalyze_pattern(bool no_perm) {
}

bool CGSolver::innerFactorize(void) {
    bool succeeded = false;
    CG.setMaxIterations(2048);
    CG.setTolerance(1e-5);
    CG.compute(coefMtr);
    succeeded = (CG.info() == Eigen::Success);
    assert(succeeded);
    return succeeded;
}

void CGSolver::innerSolve(Eigen::VectorXd &rhs, Eigen::VectorXd &result) {

    result = CG.solve(rhs);
}

void CGSolver::setParthActivation(bool activate) {
    activate_parth = activate;
}

void CGSolver::innerSolve(Eigen::MatrixXd &rhs, Eigen::MatrixXd &result) {
    result = CG.solve(rhs);
}

void CGSolver::resetSolver() {

}

LinSysSolverType CGSolver::type() const { return LinSysSolverType::CG; };
} // namespace PARTH_SOLVER

