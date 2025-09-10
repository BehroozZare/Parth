//
// Created by Behrooz on 2025-09-10.
//


#ifndef GET_FACTOR_NNZ_H
#define GET_FACTOR_NNZ_H
#include <cholmod.h>
#include <vector>
//Return the factor's nnz using CHOLMOD analysis
int get_factor_nnz(int* Ap, int* Ai, double* Ax, int N, int NNZ, std::vector<int>& perm);

#endif
