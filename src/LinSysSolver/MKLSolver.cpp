#ifdef PARTH_WITH_MKL

#include "MKLSolver.hpp"
#include "omp.h"

namespace PARTH_SOLVER {
MKLSolver::~MKLSolver() {
  phase = -1; /* Release internal memory. */
  PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &N, &Ax, Ap, Ai, NULL, &nrhs,
          iparm, &msglvl, &ddum, &ddum, &error);
}

MKLSolver::MKLSolver() {
  setMKLConfigParam();
  if (Base::reorderingType == ReorderingType::METIS)
    parth.setReorderingType(PARTH::ReorderingType::METIS);
  else if (Base::reorderingType == ReorderingType::AMD) {
    parth.setReorderingType(PARTH::ReorderingType::AMD);
  }

  Ap = nullptr;
  Ai = nullptr;
  Ax = nullptr;
  N_MKL = 0;

  parth.clearParth();
  parth.setVerbose(true);
  parth.setNDLevels(6);
  parth.setNumberOfCores(10);
  Base::initVariables();
}

void MKLSolver::setMKLConfigParam() {
  for (int i = 0; i < 64; i++) {
    pt[i] = 0;
    iparm[i] = 0;
  }

  iparm[0] = 1; /* No solver default */
  iparm[1] = 2; /* Fill-in reordering from METIS */
  iparm[2] = 0;
  iparm[3] = 0; /* No iterative-direct algorithm */
  if (activate_parth) {
    iparm[4] = 1; /* User permutation. */
  } else {
    iparm[4] = 0; /* User permutation is ignored */
  }
  iparm[5] = 0;   /* Write solution into x */
  iparm[6] = 0;   /* Not in use */
  iparm[7] = 1;   /* Max numbers of iterative refinement steps */
  iparm[8] = 0;   /* Not in use */
  iparm[9] = 0;   /* Perturb the pivot elements with 1E-8 */
  iparm[10] = 0;  /* Use nonsymmetric permutation and scaling MPS */
  iparm[11] = 0;  /* A^TX=B */
  iparm[12] = 0;  /* Maximum weighted matching algorithm is switched-off
                     (default  for symmetric). Try iparm[12] = 1 in case of
                     inappropriate  accuracy */
  iparm[13] = 0;  /* Output: Number of perturbed pivots */
  iparm[14] = 0;  /* Not in use */
  iparm[15] = 0;  /* Not in use */
  iparm[16] = 0;  /* Not in use */
  iparm[17] = -1; /* Output: Number of nonzeros in the factor LU */
  iparm[18] = -1; /* Output: Mflops for LU factorization */
  iparm[19] = 0;  /* Output: Numbers of CG Iterations */
  iparm[20] = 1;  /*using bunch kaufman pivoting*/
  iparm[55] = 0;  /*Diagonal and pivoting control., default is zero*/
  iparm[59] =
      1; /* Use in-core intel MKL pardiso if the sze of the problem do not
            exceed MKL_PARDISO_OOC_MAX_CORE_SIZE. Otherwise, use out-of-core*/
  //
  iparm[26] = 1;
  // iparm[23] = 1; //TODO: Later enable to se if the parallelism is better
  iparm[34] = 1;
  // Because iparm[4]==0 so:
  iparm[30] = 0;
  iparm[35] = 0;

  maxfct = 1; /* Maximum number of numerical factorizations. */
  mnum = 1;   /* Which factorization to use. */
  msglvl = 0; /* Print statistical information in file */
  error = 0;  /* Initialize error flag */
  nrhs = 1;   /* Number of right hand sides. */
  mtype = 2;  // Real and SPD matrices
}

void MKLSolver::setMatrix(int *p, int *i, double *x, int A_N, int NNZ, int *Mp,
                          int *Mi, int M_N, std::vector<int> &new_to_old_map) {
  if (new_to_old_map.empty()) {
    parth.setMeshPointers(M_N, Mp, Mi);
  } else {
    parth.setMeshPointers(M_N, Mp, Mi, new_to_old_map);
  }

  assert(p[A_N] == NNZ);
  this->N = A_N;
  this->NNZ = NNZ;
  this->N_MKL = A_N;

  this->Ap = p;
  this->Ai = i;
  this->Ax = x;
}

void MKLSolver::innerAnalyze_pattern(bool no_perm) {
  // Clean memory
  setMKLConfigParam();
  if(!activate_parth){
    if (reorderingType == ReorderingType::METIS) {
      iparm[1] = 2; /* Fill-in reordering from METIS */
      std::cout << "Using METIS" << std::endl;
    } else if (Base::reorderingType == ReorderingType::AMD) {
      iparm[1] = 3; /* Fill-in reordering from AMD */
      std::cout << "Using AMD" << std::endl;
    }
    iparm[4] = 0; /* User permutation is ignored */
  } else {
    iparm[4] = 1; /* User permutation. */
  }
  assert(N == N_MKL);
  assert(Ap[N_MKL] == NNZ);
  phase = -1;
  PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &N, Ax, Ap, Ai, NULL, &nrhs,
          iparm, &msglvl, &ddum, &ddum, &error);


  phase = 11;
  if (activate_parth) {
    PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &N_MKL, Ax, Ap, Ai, perm.data(),
            &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
  } else {
    PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &N_MKL, Ax, Ap, Ai, NULL, &nrhs,
            iparm, &msglvl, &ddum, &ddum, &error);
  }

  if (error != 0) {
    std::cerr << "ERROR during symbolic factorization - code: " << error
              << std::endl;
  }
}

bool MKLSolver::innerFactorize(void) {
  phase = 0;
  PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &N_MKL, Ax, Ap, Ai, NULL, &nrhs,
          iparm, &msglvl, &ddum, &ddum, &error);

  phase = 22;
  PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &N_MKL, Ax, Ap, Ai, NULL, &nrhs,
          iparm, &msglvl, &ddum, &ddum, &error);

  L_NNZ = iparm[17];
  if (error != 0) {
    std::cerr << "ERROR during numerical factorization - code: " << error
              << std::endl;
    return false;
  }
  return true; // TODO:CHECK FOR SPD FLAGS LATER
}

void MKLSolver::innerSolve(Eigen::VectorXd &rhs, Eigen::VectorXd &result) {
  double *x = (double *)mkl_calloc(rhs.size() * nrhs, sizeof(double), 64);
  phase = 33;
  iparm[7] = 0; /* Max numbers of iterative refinement steps. */

  PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &N_MKL, Ax, Ap, Ai, NULL, &nrhs,
          iparm, &msglvl, rhs.data(), x, &error);


  if (error != 0) {
    std::cerr << "ERROR during solve - code: " << error << std::endl;
  }
  result = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(x, N);
  mkl_free(x);

}

void MKLSolver::setParthActivation(bool activate) {
  activate_parth = activate;
  if (activate_parth) {
    iparm[4] = 1; /* User permutation. */
  } else {
    iparm[4] = 0; /* User permutation is ignored */
  }
}

void MKLSolver::innerSolve(Eigen::MatrixXd &rhs, Eigen::MatrixXd &result) {
  nrhs = rhs.cols();
  result.resize(rhs.rows(), rhs.cols());
  phase = 33;
  iparm[7] = 0; /* Max numbers of iterative refinement steps. */
  PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &N_MKL, Ax, Ap, Ai, NULL, &nrhs,
          iparm, &msglvl, rhs.data(), result.derived().data(), &error);

  if (error != 0) {
    std::cerr << "ERROR during solve - code: " << error << std::endl;
  }

}

void MKLSolver::resetSolver() {
  phase = -1; /* Release internal memory. */
  PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &N, Ax, Ap, Ai, NULL, &nrhs,
          iparm, &msglvl, &ddum, &ddum, &error);
  Base::resetParth();
  this->setMKLConfigParam();
  Ap = nullptr;
  Ai = nullptr;
  Ax = nullptr;
  N_MKL = 0;
}

LinSysSolverType MKLSolver::type() const { return LinSysSolverType::MKL_LIB; };
} // namespace PARTH_SOLVER

#endif
