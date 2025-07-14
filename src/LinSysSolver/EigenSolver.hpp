#include "LinSysSolver.hpp"

#include "Parth.h"

#include <Eigen/Eigen>
#include <Eigen/SparseCholesky>

#include <set>
#include <vector>

namespace PARTH_SOLVER {

class EigenSolver : public LinSysSolver {
  typedef LinSysSolver Base; // The class
public:                // Access specifier
    Eigen::SparseMatrix<double> coefMtr;
       Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> simplicialLDLT;

  ~EigenSolver();
   EigenSolver();


  void setMatrix(int *p, int *i, double *x, int A_N, int NNZ, int *Mp, int *Mi,
                 int M_N, std::vector<int> &new_to_old_map) override;
  void innerAnalyze_pattern(bool no_perm) override;
  bool innerFactorize(void) override;
  void innerSolve(Eigen::VectorXd &rhs, Eigen::VectorXd &result) override;
  void innerSolve(Eigen::MatrixXd &rhs, Eigen::MatrixXd &result) override;
  void resetSolver() override;
  virtual LinSysSolverType type() const override;

  virtual void setParthActivation(bool activate) override;

};

}