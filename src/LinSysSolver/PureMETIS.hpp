#include "LinSysSolver.hpp"

#include "Parth.h"

#include <Eigen/Eigen>

#include <set>
#include <vector>

#include <cholmod.h>

namespace PARTH_SOLVER {

class PureMETIS : public LinSysSolver {
  typedef LinSysSolver Base; // The class
public:                // Access specifier
    Eigen::SparseMatrix<double> coefMtr;
    Eigen::SparseMatrix<double> fullMtr;
	Eigen::SparseMatrix<double> M_Comp;
	cholmod_common cm;
	cholmod_sparse *A;
	cholmod_factor *L;
	cholmod_dense *b;

	cholmod_dense *x_solve;

	void *Ai, *Ap, *Ax, *bx;


	int M_n;
	int* Mp;
	int* Mi;


  ~PureMETIS();
  PureMETIS();


  void setMatrix(int *p, int *i, double *x, int A_N, int NNZ, int *Mp, int *Mi,
                 int M_N, std::vector<int> &new_to_old_map) override;
  void innerAnalyze_pattern(bool no_perm) override;
  bool innerFactorize(void) override;
  void innerSolve(Eigen::VectorXd &rhs, Eigen::VectorXd &result) override;
  void innerSolve(Eigen::MatrixXd &rhs, Eigen::MatrixXd &result) override;
  void resetSolver() override;
  virtual LinSysSolverType type() const override;
	void cholmod_clean_memory();
  virtual void setParthActivation(bool activate) override;

};

}