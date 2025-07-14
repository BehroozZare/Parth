//
// Created by behrooz zare on 2024-04-25.
//

#ifndef PARTH_PROJECTED_COTMATRIX_H
#define PARTH_PROJECTED_COTMATRIX_H

#include <Eigen/Dense>
#include <Eigen/Sparse>


namespace PARTHDEMO {
    /// Constructs the cotangent stiffness matrix (discrete laplacian) for a given
    /// mesh (V,F).
    ///
    ///   @tparam DerivedV  derived type of eigen matrix for V (e.g. derived from
    ///     MatrixXd)
    ///   @tparam DerivedF  derived type of eigen matrix for F (e.g. derived from
    ///     MatrixXi)
    ///   @tparam Scalar  scalar type for eigen sparse matrix (e.g. double)
    ///   @param[in] V  #V by dim list of mesh vertex positions
    ///   @param[in] F  #F by simplex_size list of mesh elements (triangles or tetrahedra)
    ///   @param[out] L  #V by #V cotangent matrix, each row i corresponding to V(i,:)
    ///
    /// \see
    ///   adjacency_matrix
    ///
    /// \note This Laplacian uses the convention that diagonal entries are
    /// **minus** the sum of off-diagonal entries. The diagonal entries are
    /// therefore in general negative and the matrix is **negative** semi-definite
    /// (immediately, -L is **positive** semi-definite)
    ///
    void projected_cotmatrix(
            const Eigen::MatrixXd & V,
            const Eigen::MatrixXi & F,
            Eigen::SparseMatrix<double>& L);

}


#endif //PARTH_SOLVER_DECIMATION_H
