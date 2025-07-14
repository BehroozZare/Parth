//
// Created by behrooz zare on 2024-04-25.
//

#ifndef PARTH_SOLVER_REMESH_H
#define PARTH_SOLVER_REMESH_H

#include <Eigen/Core>
#include <vector>


namespace PARTHDEMO {
    Eigen::VectorXi remesh(Eigen::VectorXi &I, Eigen::MatrixXi &F,
                          Eigen::MatrixXd &V, Eigen::MatrixXi &F_new,
                          Eigen::MatrixXd &V_new, double scale, std::vector<int>& new_to_old_dof_map);
}


#endif //PARTH_SOLVER_DECIMATION_H
