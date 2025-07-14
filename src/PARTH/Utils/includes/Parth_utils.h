//
// Created by behrooz on 2022-04-26.
//

#ifndef PARTH_UTILS_H
#define PARTH_UTILS_H

#include "csv_utils.h"
#include <Eigen/Eigen>
#include <Eigen/Sparse>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <tuple>
#include <vector>
#include <cassert>


namespace PARTH {

    void printEigenMatrix(Eigen::MatrixXd &V, long max_row_cnt = 10);

    void printEigenMatrix(Eigen::VectorXd &V, long max_row_cnt = 10);

    void printEigenMatrix(Eigen::VectorXi &V, long max_row_cnt = 10);

    Eigen::VectorXd flattenVector(const Eigen::MatrixXd &x);

///---------------------------------------------------------------------------------------\n
/// convertCSCintoEigenSparse - Convert the CSC format matrix into Eigen Sparse
/// matrix
///---------------------------------------------------------------------------------------\n
    template<typename T1, typename T2>
    void convertCompressFormatintoEigenSparse(
            int n,                              ///<[in] number of rows/columns
            int nnz,                            ///<[in] number of nnz
            const T2 *Ap,                       ///<[in] pointer array in CSC format
            const T2 *Ai,                       ///<[in] index array in CSC format
            const T1 *Ax,                       ///<[in] value array in CSC format
            Eigen::SparseMatrix<T1> &eigen_mat, ///<[out] eigen sparse matrix as output
            bool makefull = false ///<[in] if the input is lower triangular, it will
            ///< convert it into full matrix
    ) {
        assert(n > 0 && nnz > 0);
        assert(Ap != nullptr);
        assert(Ai != nullptr);
        assert(Ax != nullptr);

        if (makefull) {
            eigen_mat.resize(n, n);
            eigen_mat.reserve(nnz * 2 - n);

            for (auto col = 0; col < n; col++) {
                for (auto row_ptr = Ap[col]; row_ptr < Ap[col + 1]; row_ptr++) {
                    auto row = Ai[row_ptr];
                    T1 val = Ax[row_ptr];
                    eigen_mat.insert(row, col) = val;
                    if (row != col) {
                        eigen_mat.insert(col, row) = val;
                    }
                }
            }
        } else {
            eigen_mat.conservativeResize(n, n);
            eigen_mat.reserve(nnz);

            std::copy(Ap, Ap + (n + 1), eigen_mat.outerIndexPtr());
            std::copy(Ai, Ai + nnz, eigen_mat.innerIndexPtr());
            std::copy(Ax, Ax + nnz, eigen_mat.valuePtr());
        }
    }

///---------------------------------------------------------------------------------------\n
/// Synthesize 3D Possion PDE problem\n
///---------------------------------------------------------------------------------------\n
    void compute3DPossion(Eigen::SparseMatrix<double, Eigen::RowMajor>
                          &A,    ///<[in/out] The input matrix in CSR format
                          int n = 30 ///<[in] The number of DOFs in each direction
    );

///---------------------------------------------------------------------------------------\n
/// Compute the mesh from the Hessian structure
///---------------------------------------------------------------------------------------\n
Eigen::SparseMatrix<double> computeMeshFromHessian(Eigen::SparseMatrix<double>
                          &A,    ///<[in/out] The input matrix in CSC format
                          int DIM=3 ///[in] The dimension of the hessian
);


///---------------------------------------------------------------------------------------\n
/// create a list of files in a folder.
///---------------------------------------------------------------------------------------\n
/// @return A list of names of files in a folder
    std::vector<std::string>
    getFileNames(std::string path ///<[in] the address of the folder
    );

///---------------------------------------------------------------------------------------\n
/// split a string based on the delimiter\n
///---------------------------------------------------------------------------------------\n
/// @return a vector of string\n
    std::vector<std::string>
    split_string(std::string input,    ///<[in] the address of the folder
                 std::string delimiter ///<[in] the address of the folder
    );

///---------------------------------------------------------------------------------------\n
/// comparison operator by first and second element of a tuple\n
///---------------------------------------------------------------------------------------\n
    bool sortbyFirstAndSec(const std::tuple<int, int> &a,
                           const std::tuple<int, int> &b);

///---------------------------------------------------------------------------------------\n
/// Compute dot product of two vector without overflow
///---------------------------------------------------------------------------------------\n
    double dotProduct(double *a, double *b, int size);

///---------------------------------------------------------------------------------------\n
/// Compute the summation of two vector in the form of result = a_scale * a +
/// b_scale * b (NOTE: result can be equal to one of the a and b)
///---------------------------------------------------------------------------------------\n
    void vectorSum(
            double *a,      ///<[in] input a
            double a_scale, ///<[in] input a_scale * a
            double *b,      ///<[in] input b
            double b_scale, ///<[in] input b_scale * b
            int size,       ///<[in] size of the a or b (they should be equal)
            double *result  ///<[out] result = result = a_scale * a + b_scale * b
    );

///---------------------------------------------------------------------------------------\n
/// Compute the inf norm of matrix A in CSC/CSR format (max(A[i]) for 0 <= i < n
///---------------------------------------------------------------------------------------\n
    double
    computeInfNormOfMatrixA(double *Ax, ///<[in] Value vector in CSC/CSR format
                            int size    ///[in] nnz in A
    );

///---------------------------------------------------------------------------------------\n
/// Compute norm 2 of two vector without overflow
///---------------------------------------------------------------------------------------\n
    double Norm2(Eigen::VectorXi &a);

///---------------------------------------------------------------------------------------\n
/// Load a dense matrix from CSV file
///---------------------------------------------------------------------------------------\n
    Eigen::MatrixXi
    openMatInt(std::string fileToOpen /// <[in] input address for the file
    );

///---------------------------------------------------------------------------------------\n
/// Load a dense matrix from CSV file
///---------------------------------------------------------------------------------------\n
    Eigen::MatrixXd openMatDouble(std::string fileToOpen/// <[in] input address for the file
  );
///---------------------------------------------------------------------------------------\n
/// save a dense matrix to the CSV file
///---------------------------------------------------------------------------------------\n
    void saveMat(std::string fileName,  /// <[in] output address
                 Eigen::MatrixXd matrix /// <[out] Dense matrix
    );

///---------------------------------------------------------------------------------------\n
/// give the index of biggest values
///---------------------------------------------------------------------------------------\n
    std::vector<int> getBiggestIds(std::vector<double> &input // The input array
    );

///---------------------------------------------------------------------------------------\n
/// find a variable inside the vector
///---------------------------------------------------------------------------------------\n
    template<typename T>
    bool isInVector(const T &what, const std::vector<T> &vec) {
        bool result = (std::find(vec.begin(), vec.end(), what) != vec.end());
        return result;
    }







    void print_csc(
            size_t n,                                ///<[in] Number of nodes
            int *Ap,                                 ///<[in] pointer array
            int *Ai,                                 ///<[in] index array
            double *Ax,                              ///<[in] value array
            std::streambuf *out = std::cout.rdbuf(), ///<[in] output buffer
            const std::string indent = "  ",         ///<[in] indent definition
            const std::string &beg =
            "%%MatrixMarket matrix coordinate real symmetric" ///<[in] Header of the
            ///< MMarket format
    );

///---------------------------------------------------------------------------------------\n
/// Write multiple obj files based on the total V and SF. The Vs within a single group are
/// written to a single file
///---------------------------------------------------------------------------------------\n
    bool WriteObjWithGroup(Eigen::MatrixXd V, ///[in] The nodal positions
                           Eigen::MatrixXi F, /// [in] Faces
                           std::vector<int> group, /// [in] a mapping from node to group
                           std::vector<std::string>, /// [in] A mapping from group to its name used for saving obj files
                           std::string base_address /// The folder in which the obj files are saved
    );

    ///---------------------------------------------------------------------------------------\n
    /// Write multiple obj files based on the total V and SF. The Vs within a single group are
    /// written to a single file
    ///---------------------------------------------------------------------------------------\n
    bool applyTextureBasedOnLeavesAndSeparator(Eigen::MatrixXd V, ///[in] The nodal positions
                           Eigen::MatrixXi SF, /// [in] Surface Faces
                           int max_lvl, ///[in] max_level
                           std::vector<int> group, /// [in] a mapping from node to group
                           std::string base_address,/// The folder in which the obj files are saved
                           int frame, int iter
    );


} // namespace PARTH

#endif // IPC_PARTH_UTILS_H
