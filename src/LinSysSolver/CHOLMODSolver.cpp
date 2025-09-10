//
//  CHOLMODSolver.cpp
//  IPC
//
//  Created by Minchen Li on 6/22/18.
//

#ifdef PARTH_WITH_CHOLMOD

#include "CHOLMODSolver.hpp"
#include "omp.h"

namespace PARTH_SOLVER {
CHOLMODSolver::~CHOLMODSolver() {
  if (A) {
    A->i = Ai;
    A->p = Ap;
    A->x = Ax;
    cholmod_free_sparse(&A, &cm);
  }

  cholmod_free_factor(&L, &cm);

  if (b) {
    b->x = bx;
    cholmod_free_dense(&b, &cm);
  }

  if (x_solve) {
    cholmod_free_dense(&x_solve, &cm);
  }

  cholmod_finish(&cm);
}

CHOLMODSolver::CHOLMODSolver() {
  cholmod_start(&cm);
  A = NULL;
  L = NULL;
  b = NULL;
  x_solve = NULL;
  Ai = Ap = Ax = NULL;

  this->activate_parth = activate_parth;

  if (activate_parth) {
    std::cout << "Activating Immobilizer" << std::endl;
  } else {
    std::cout << "Deactivating Immobilizer" << std::endl;
  }

  parth.setVerbose(true);
  parth.setNDLevels(6);
  parth.setNumberOfCores(10);
}

void CHOLMODSolver::cholmod_clean_memory() {
  if (A) {
    A->i = Ai;
    A->p = Ap;
    A->x = Ax;
    cholmod_free_sparse(&A, &cm);
  }

  if (b) {
    b->x = bx;
    cholmod_free_dense(&b, &cm);
  }

  if (x_solve) {
    cholmod_free_dense(&x_solve, &cm);
  }

  A = NULL;
  b = NULL;
  x_solve = NULL;
  Ai = Ap = Ax = NULL;
}

void CHOLMODSolver::setMatrix(int *p, int *i, double *x, int A_N, int NNZ,
                              int *Mp, int *Mi, int M_N,
                              std::vector<int> &new_to_old_map) {
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

  this->cholmod_clean_memory();

  if (!A) {
    A = cholmod_allocate_sparse(N, N, NNZ, true, true, -1, CHOLMOD_REAL, &cm);
    this->Ap = A->p;
    this->Ax = A->x;
    this->Ai = A->i;
    // -1: upper right part will be ignored during computation
  }

  A->p = p;
  A->i = i;
  A->x = x;
}

void CHOLMODSolver::innerAnalyze_pattern(bool no_perm) {
  cholmod_free_factor(&L, &cm);

  cm.supernodal = CHOLMOD_SUPERNODAL;
  if (activate_parth) {
    cm.nmethods = 1;
    cm.method[0].ordering = CHOLMOD_GIVEN;
    L = cholmod_analyze_p(A, perm.data(), NULL, 0, &cm);
  } else {
    cm.nmethods = 1;
    if (Base::reorderingType == ReorderingType::METIS) {
      cm.method[0].ordering = CHOLMOD_METIS;
      std::cout << "*** CHOLMOD: Choosing METIS" << std::endl;
    } else if (Base::reorderingType == ReorderingType::AMD) {
      cm.method[0].ordering = CHOLMOD_AMD;
      std::cout << "*** CHOLMOD: Choosing AMD" << std::endl;
    } else {
      std::cerr << "*** CHOLMOD: UNKNOWN REORDERING TYPE -> Choosing METIS"
                << std::endl;
    };
    L = cholmod_analyze(A, &cm);
  }

//    if (cm.selected == 1) {
//        std::cout << "The method is AMD." << std::endl;
//        Base::reorderingType = ReorderingType::AMD;
//    } else if (cm.selected == 2) {
//        std::cout << "The method is METIS." << std::endl;
//        Base::reorderingType = ReorderingType::METIS;
//    } else if (cm.selected == 0) {
//        std::cout << "The meth"
//                     "od is UserDefined."
//                  << std::endl;
//    } else {
//        std::cout << "The method " << cm.selected << " is unknown"
//        << std::endl;
//    }

  L_NNZ = cm.lnz * 2 - N;

  if (L == nullptr) {
    std::cerr << "ERROR during symbolic factorization:" << std::endl;
  }
}

bool CHOLMODSolver::innerFactorize(void) {
  cholmod_factorize(A, L, &cm);
  if (cm.status == CHOLMOD_NOT_POSDEF) {
    std::cerr << "ERROR during numerical factorization - code: " << std::endl;
    assert(true);
    return false;
  }
  return true; // TODO:CHECK FOR SPD FLAGS LATER
}

void CHOLMODSolver::innerSolve(Eigen::VectorXd &rhs, Eigen::VectorXd &result) {
  if (!b) {
    b = cholmod_allocate_dense(N, 1, N, CHOLMOD_REAL, &cm);
    bx = b->x;
  }
  b->x = rhs.data();

  if (x_solve) {
    cholmod_free_dense(&x_solve, &cm);
  }

  x_solve = cholmod_solve(CHOLMOD_A, L, b, &cm);

  result.conservativeResize(rhs.size());
  memcpy(result.data(), x_solve->x, result.rows() * sizeof(result[0]));
  double one[2] = {1, 0}, m1[2] = {-1, 0};
  cholmod_dense *chol_r = cholmod_copy_dense(b, &cm);
  cholmod_sdmult(A, 0, m1, one, x_solve, chol_r, &cm);
  residual = cholmod_norm_dense(chol_r, 0, &cm);
  std::cout << "CHOLMOD_res " << residual << std::endl;
  cholmod_free_dense(&chol_r, &cm);
}

void CHOLMODSolver::innerSolve(Eigen::MatrixXd &rhs, Eigen::MatrixXd &result) {
  result.resize(rhs.rows(), rhs.cols());
  if (!b || b->nrow != rhs.rows() || b->ncol != rhs.cols()) {
    b = cholmod_allocate_dense(rhs.rows(), rhs.cols(), rhs.rows(), CHOLMOD_REAL,
                               &cm);
    bx = b->x;
    if (b->nrow != N) {
      std::cerr << "ERROR: rhs dimension is not equal to matrix dimension"
                << std::endl;
    }
  }

  b->x = rhs.data();

  if (x_solve) {
    cholmod_free_dense(&x_solve, &cm);
  }

  x_solve = cholmod_solve(CHOLMOD_A, L, b, &cm);

  memcpy(result.data(), x_solve->x,
         result.rows() * result.cols() * sizeof(result[0]));
}

void CHOLMODSolver::resetSolver() {
  Base::resetParth();
  cholmod_clean_memory();

  A = NULL;
  L = NULL;
  b = NULL;
  x_solve = NULL;
  Ai = Ap = Ax = NULL;
  bx = NULL;
}

LinSysSolverType CHOLMODSolver::type() const {
  return LinSysSolverType::CHOLMOD;
};
} // namespace PARTH_SOLVER

#endif
