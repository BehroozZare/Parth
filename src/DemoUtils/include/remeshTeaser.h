//
// Created by behrooz zare on 2024-04-25.
//

#ifndef PARTH_SOLVER_REMESH_H
#define PARTH_SOLVER_REMESH_H

#include <Eigen/Core>
#include <vector>


namespace PARTHDEMO {
    Eigen::VectorXi remeshTeaser(Eigen::VectorXi &I, ///<[in] The set of indices of faces to be remeshed
                                 Eigen::MatrixXi &F, ///<[in] Total triangle mesh faces
                          Eigen::MatrixXd &V, ///<[in] Total triangle mesh vertices
                          Eigen::MatrixXi &F_new, ///<[out] The new faces after remeshing
                          Eigen::MatrixXd &V_new, ///<[out] The new vertices after remeshing
                          double scale, ///<[in] The scale of the remeshing used for botsch remesher
                          std::vector<int>& new_to_old_dof_map ///<[out] The mapping of new vertices to old vertices
                          );
}


#endif //PARTH_SOLVER_DECIMATION_H
