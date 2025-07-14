#ifdef PARTH_WITH_MKL

#include "LinSysSolver.hpp"

#include "Parth.h"

#include <mkl.h>

#include <Eigen/Eigen>

#include <set>
#include <vector>

namespace PARTH_SOLVER {

class MKLSolver : public LinSysSolver {
  typedef LinSysSolver Base; // The class
public:                // Access specifier
  MKL_INT mtype = 2; /* Real unsymmetric matrix */
  MKL_INT nrhs = 1;  /* Number of right hand sides. */
  long int pt[64];
  /* Pardiso control parameters. */
  MKL_INT iparm[64];
  MKL_INT maxfct, mnum, phase, error, msglvl;
  /* Auxiliary variables. */
  double ddum;  /* Double dummy */
  MKL_INT idum; /* Integer dummy. */

  MKL_INT *Ap;
  MKL_INT *Ai;
  double *Ax;
  MKL_INT N_MKL;

  ~MKLSolver();
  MKLSolver();

  void setMatrix(int *p, int *i, double *x, int A_N, int NNZ, int *Mp, int *Mi,
                 int M_N, std::vector<int> &new_to_old_map) override;
  void innerAnalyze_pattern(bool no_perm) override;
  bool innerFactorize(void) override;
  void innerSolve(Eigen::VectorXd &rhs, Eigen::VectorXd &result) override;
  void innerSolve(Eigen::MatrixXd &rhs, Eigen::MatrixXd &result) override;
  void resetSolver() override;
  void setMKLConfigParam();
  virtual LinSysSolverType type() const override;

  virtual void setParthActivation(bool activate) override;

};

}

#endif