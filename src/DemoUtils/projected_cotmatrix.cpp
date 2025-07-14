//
// Created by behrooz zare on 2024-04-25.
//

#include "projected_cotmatrix.h"
#include <igl/cotmatrix_entries.h>

namespace PARTHDEMO {
    void MakeSymmetricMatrixSPD(Eigen::MatrixXd& matrix) {
        // Assume the matrix is symmetric
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(matrix);

        // Get eigenvalues and clamp any negative values to zero
        Eigen::VectorXd eigenValues = eigensolver.eigenvalues();
        bool projected = false;
        for (int i = 0; i < eigenValues.rows(); ++i) {
            if(eigenValues[i] < 0){
                eigenValues[i] = -eigenValues[i];
                projected = true;
            }
        }

        // Recreate the matrix as P * D * P'
        if(projected){
            matrix = eigensolver.eigenvectors() * eigenValues.asDiagonal() * eigensolver.eigenvectors().transpose();
        }

    }

    void projected_cotmatrix(
            const Eigen::MatrixXd & V,
            const Eigen::MatrixXi & F,
            Eigen::SparseMatrix<double>& L)
    {
        using namespace Eigen;
        using namespace std;

        L.resize(V.rows(),V.rows());
        Matrix<int,Dynamic,2> edges;
        int simplex_size = F.cols();
        // 3 for triangles, 4 for tets
        assert(simplex_size == 3 || simplex_size == 4);
        if(simplex_size == 3)
        {
            // This is important! it could decrease the comptuation time by a factor of 2
            // Laplacian for a closed 2d manifold mesh will have on average 7 entries per
            // row
            L.reserve(10*V.rows());
            edges.resize(3,2);
            edges <<
                  1,2,
                    2,0,
                    0,1;
        }else if(simplex_size == 4)
        {
            L.reserve(17*V.rows());
            edges.resize(6,2);
            edges <<
                  1,2,
                    2,0,
                    0,1,
                    3,0,
                    3,1,
                    3,2;
        }else
        {
            return;
        }
        // Gather cotangents
        Eigen::MatrixXd C;
        igl::cotmatrix_entries(V,F,C);

        vector<Triplet<double> > IJV;
        IJV.reserve(F.rows()*edges.rows()*4);
        // Loop over triangles
        Eigen::MatrixXd C_matrix(3,3);
        for(int i = 0; i < F.rows(); i++)
        {
            if(simplex_size == 3){
                //Creating the symetric matrix from C and projecting it

                C_matrix.setZero();
                for(int e = 0;e<edges.rows();e++)
                {
                    int source = edges(e,0);
                    int dest = edges(e,1);
                    C_matrix(source,dest) += -C(i,e);
                    C_matrix(dest,source) += -C(i,e);
                    C_matrix(source,source) += C(i,e);
                    C_matrix(dest,dest) += C(i,e);
                }
                MakeSymmetricMatrixSPD(C_matrix);
                //Add to triplet IJV
                for(int j = 0; j < C_matrix.rows(); j++)
                    for(int k = 0; k < C_matrix.cols(); k++)
                        IJV.push_back(Triplet<double>(F(i,j),F(i,k),-C_matrix(j,k)));
            } else {
                // loop over edges of element
                for(int e = 0;e<edges.rows();e++)
                {
                    int source = F(i,edges(e,0));
                    int dest = F(i,edges(e,1));
                    IJV.push_back(Triplet<double>(source,dest,C(i,e)));
                    IJV.push_back(Triplet<double>(dest,source,C(i,e)));
                    IJV.push_back(Triplet<double>(source,source,-C(i,e)));
                    IJV.push_back(Triplet<double>(dest,dest,-C(i,e)));
                }
            }
        }
        L.setFromTriplets(IJV.begin(),IJV.end());
    }
}