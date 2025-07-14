//
// Created by behrooz on 1/17/21.
//
#ifndef SPMP_INTERFACE_H
#define SPMP_INTERFACE_H
#include <LevelSchedule.hpp>
#include <synk/barrier.hpp>

#include "FusionDemo.h"
#include "sparse_blas_lib.h"

namespace sym_lib
{
    ///\brief convert a LBC CSR format to SpMP CSR format
    ///Note that the LBC_A should be full
    void Convert_LBCCSR_to_SpMP(CSR* LBC_A, SpMP::CSR* SpMP_A){
        auto n = LBC_A->n;
        auto nnz = LBC_A->nnz;
        // Allocate space and copy data
        SpMP_A->alloc(LBC_A->n, LBC_A->nnz, true);
        SpMP_A->n = n;
        std::copy(LBC_A->p, LBC_A->p + n + 1, SpMP_A->rowptr);
        std::copy(LBC_A->i, LBC_A->i + nnz, SpMP_A->colidx);
        std::copy(LBC_A->x, LBC_A->x + nnz, SpMP_A->values);

        // Initialize meta data for SpMP_A
        int base = 0;
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            for (int j = SpMP_A->rowptr[i] - base; j < SpMP_A->rowptr[i + 1] - base; ++j) {
                if (SpMP_A->colidx[j] - base == i) {
                    SpMP_A->diagptr[i] = j + base;
                    SpMP_A->idiag[i] = 1 / SpMP_A->values[j];
                    SpMP_A->diag[i] = SpMP_A->values[j];
                }
            }
        }
    }
}

#endif
