#include "PureMETIS.hpp"
#include <metis.h>
#include "Parth_utils.h"

namespace PARTH_SOLVER {
	PureMETIS::~PureMETIS() {
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

	PureMETIS::PureMETIS() {
		cholmod_start(&cm);
		A = NULL;
		L = NULL;
		b = NULL;
		x_solve = NULL;
		Ai = Ap = Ax = NULL;

		parth.clearParth();
		parth.setVerbose(true);
		parth.setNDLevels(6);
		parth.setNumberOfCores(10);
		Base::initVariables();
	}


	void PureMETIS::setMatrix(int *p, int *i, double *x, int A_N, int NNZ, int *Mp,
	                          int *Mi, int M_N, std::vector<int> &new_to_old_map) {
		Eigen::Map<Eigen::SparseMatrix<double> > A_lower(A_N, A_N, NNZ, p, // read-write
		                                                 i, x);
		coefMtr = A_lower;

		std::vector<Eigen::Triplet<double> > triplets;
		for (int col = 0; col < A_N; col++) {
			for (int idx = p[col]; idx < p[col + 1]; idx++) {
				triplets.push_back(Eigen::Triplet<double>(col, i[idx], x[idx]));
				if (i[idx] != col) {
					triplets.push_back(Eigen::Triplet<double>(i[idx], col, x[idx]));
				}
			}
		}
		// The trick is to add the transpose of the lower triangular part and then remove the extra diagonal.
		fullMtr.resize(A_N, A_N);
		fullMtr.setFromTriplets(triplets.begin(), triplets.end());
		fullMtr.makeCompressed();
		this->N = A_lower.rows();
		this->NNZ = A_lower.nonZeros();

		this->M_n = M_N;
		this->Mp = Mp;
		this->Mi = Mi;
		if (compression_mode != 4) {
			parth.setMeshPointers(M_N, Mp, Mi);
		} else {
			M_Comp = PARTH::computeMeshFromHessian(coefMtr, 3);
			parth.setMeshPointers(M_Comp.rows(), M_Comp.outerIndexPtr(), M_Comp.innerIndexPtr());
		}



		//============================================================
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

	void PureMETIS::innerAnalyze_pattern(bool no_perm) {
		parth_time = omp_get_wtime();
		if (compression_mode == 4) {
			std::cout << "Using PARTH with Compression" << std::endl;
			parth.setReorderingType(PARTH::ReorderingType::METIS);
			parth.computePermutation(perm, 3);
		} else if (compression_mode == 3) {
			std::cout << "Using PARTH without Compression" << std::endl;
			parth.setReorderingType(PARTH::ReorderingType::METIS);
			parth.computePermutation(perm, 1);
		} else if (compression_mode == 2) {
			//Just a disgusting way of giving reviewers what they want -> honestly I am tired of this shit
			std::cout << "Using PARTH Compression" << std::endl;
			Eigen::SparseMatrix<double> mesh_compressed = PARTH::computeMeshFromHessian(coefMtr, 3);
			//Use METIS
			idx_t N_metis = mesh_compressed.rows();
			int* Mcp = mesh_compressed.outerIndexPtr();
			int* Mci = mesh_compressed.innerIndexPtr();
			std::vector<int> tmp(N_metis);
			std::vector<int> perm_compress(N_metis);

			idx_t options[METIS_NOPTIONS];
			METIS_SetDefaultOptions(options);

			//Compression options
			// options[METIS_OPTION_COMPRESS] = 1; // Enable compression (0: disabled)

			METIS_NodeND(&N_metis, Mcp, Mci,
			             NULL, options, perm_compress.data(), tmp.data());

			//Expand the permutation
			perm.resize(coefMtr.rows());

			for (int idx = 0; idx < N_metis; idx++) {
				for (int d = 0; d < 3; d++) {
					perm[idx * 3 + d] = perm_compress[idx] * 3 + d;
				}
			}
		} else if (compression_mode == 1) {
			//Use METIS
			std::cout << "Using METIS Compression" << std::endl;
			idx_t N_metis = M_n;
			idx_t NNZ_metis = Mp[M_n];
			assert(M_n != 0);
			std::vector<int> tmp(M_n);
			std::vector<int> perm_compress(N_metis);

			idx_t options[METIS_NOPTIONS];
			METIS_SetDefaultOptions(options);

			//Compression options
			options[METIS_OPTION_COMPRESS] = 1; // Enable compression (0: disabled)
			perm.resize(N_metis);
			METIS_NodeND(&N_metis, Mp, Mi,
			             NULL, options, perm.data(), tmp.data());
		} else if (compression_mode == 0){
			//Use METIS
			std::cout << "No Compression" << std::endl;
			idx_t N_metis = M_n;
			idx_t NNZ_metis = Mp[M_n];
			assert(M_n != 0);
			std::vector<int> tmp(M_n);
			std::vector<int> perm_compress(N_metis);

			idx_t options[METIS_NOPTIONS];
			METIS_SetDefaultOptions(options);

			//Compression options
			options[METIS_OPTION_COMPRESS] = 0; // Enable compression (0: disabled)
			perm.resize(N_metis);
			METIS_NodeND(&N_metis, Mp, Mi,
				     NULL, options, perm.data(), tmp.data());
		} else {
			std::cout << "Compression mode is not supported" << std::endl;
		}
		parth_time = omp_get_wtime() - parth_time;
		cm.nmethods = 1;

		cm.method[0].ordering = CHOLMOD_GIVEN;
		L = cholmod_analyze_p(A, perm.data(), NULL, 0, &cm);
		L_NNZ = cm.lnz * 2 - N;
		std::cout << "L_NNZ: " << L_NNZ << std::endl;
	}

	bool PureMETIS::innerFactorize(void) {
		volatile int cnt = 0;
		for (int i = 0; i < 1000; i++) {
			cnt++;
		}
		return true;
	}

	void PureMETIS::innerSolve(Eigen::VectorXd &rhs, Eigen::VectorXd &result) {
	}

	void PureMETIS::setParthActivation(bool activate) {
		activate_parth = activate;
	}

	void PureMETIS::innerSolve(Eigen::MatrixXd &rhs, Eigen::MatrixXd &result) {
		rhs = result;
	}

	void PureMETIS::cholmod_clean_memory() {
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

	void PureMETIS::resetSolver() {
	}

	LinSysSolverType PureMETIS::type() const { return LinSysSolverType::PURE_METIS; };
} // namespace PARTH_SOLVER
