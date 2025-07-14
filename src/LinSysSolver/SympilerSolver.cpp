//
//  CHOLMODSolver.cpp
//  IPC
//
//  Created by Minchen Li on 6/22/18.
//

#ifdef PARTH_WITH_SYMPILER

#include "SympilerSolver.hpp"
#include "Barb.h"
#include "omp.h"
#include <thread>

namespace PARTH_SOLVER {
        SympilerSolver::~SympilerSolver() {
                delete A;
                delete sym_chol;
        }

        SympilerSolver::SympilerSolver() {
                A = new sym_lib::parsy::CSC;
                sym_chol = NULL;

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

        void SympilerSolver::setMatrix(int *p, int *i, double *x, int A_N, int NNZ,
                                       int *Mp, int *Mi, int M_N,
                                       std::vector<int> &new_to_old_map) {
                assert(p[A_N] == NNZ);
                this->N = A_N;
                this->NNZ = NNZ;

                if (new_to_old_map.empty()) {
                        parth.setMeshPointers(M_N, Mp, Mi);
                } else {
                        parth.setMeshPointers(M_N, Mp, Mi, new_to_old_map);
                }

                if (A == nullptr) {
                        A = new sym_lib::parsy::CSC;
                }
                A->i = i;
                A->p = p;
                A->x = x;
                A->nzmax = NNZ;
                A->ncol = A->nrow = A_N;
                A->packed = 1;
                A->sorted = 1;
                A->stype = -1; // lower triangular
        }

        void SympilerSolver::innerAnalyze_pattern(bool no_perm) {
                if (activate_parth) {
                        std::cerr << "sympiler does not permit custom fill-ins" << std::endl;
                } else {
                        if (sym_chol) {
                                delete sym_chol;
                        }

                        const auto processor_count = std::thread::hardware_concurrency() / 2;
                        sym_chol = sympiler::sympiler_chol_symbolic(A, processor_count);
                }

                L_NNZ = -1;
        }

        bool SympilerSolver::innerFactorize(void) {
                bool ret = sym_chol->numerical_factorization(A);
                return ret;
        }

        void SympilerSolver::innerSolve(Eigen::VectorXd &rhs, Eigen::VectorXd &result) {
                result.conservativeResize(rhs.size());
                auto *sol_dbl = sym_chol->solve_only(rhs.data(), 1);
                result.conservativeResize(rhs.size());
                memcpy(result.data(), sol_dbl, rhs.size() * sizeof(double));
        }

        void SympilerSolver::innerSolve(Eigen::MatrixXd &rhs, Eigen::MatrixXd &result) {
                std::cerr << "Multiple right hand side is not supported" << std::endl;
        }

        void SympilerSolver::resetSolver() {
                Base::resetParth();
                delete sym_chol;
                sym_chol = nullptr;
        }

        LinSysSolverType SympilerSolver::type() const {
                return LinSysSolverType::SYMPILER;
        };
} // namespace PARTH_SOLVER

#endif
