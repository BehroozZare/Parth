

#include "EigenSolver.hpp"


namespace PARTH_SOLVER {
EigenSolver::~EigenSolver() {

}

EigenSolver::EigenSolver() {

  parth.clearParth();
  parth.setVerbose(true);
  parth.setNDLevels(6);
  parth.setNumberOfCores(10);
  Base::initVariables();
}


void EigenSolver::setMatrix(int *p, int *i, double *x, int A_N, int NNZ, int *Mp,
                          int *Mi, int M_N, std::vector<int> &new_to_old_map) {
        this->N = A_N;
        this->NNZ = NNZ;
        Eigen::Map<Eigen::SparseMatrix<double> > A_lower(A_N,A_N,NNZ,p, // read-write
                                                        i,x);
        coefMtr = A_lower;
}

void EigenSolver::innerAnalyze_pattern(bool no_perm) {
        simplicialLDLT.analyzePattern(coefMtr);
        assert(simplicialLDLT.info() == Eigen::Success);
}

bool EigenSolver::innerFactorize(void) {
        bool succeeded = false;
        simplicialLDLT.factorize(coefMtr);
        succeeded = (simplicialLDLT.info() == Eigen::Success);
        assert(succeeded);
        return succeeded;
}

void EigenSolver::innerSolve(Eigen::VectorXd &rhs, Eigen::VectorXd &result) {
        result = simplicialLDLT.solve(rhs);
        assert(simplicialLDLT.info() == Eigen::Success);
}

void EigenSolver::setParthActivation(bool activate) {
    activate_parth = activate;
}

void EigenSolver::innerSolve(Eigen::MatrixXd &rhs, Eigen::MatrixXd &result) {
        result = simplicialLDLT.solve(rhs);
        assert(simplicialLDLT.info() == Eigen::Success);
}

void EigenSolver::resetSolver() {

}

LinSysSolverType EigenSolver::type() const { return LinSysSolverType::EIGEN; };
} // namespace PARTH_SOLVER

