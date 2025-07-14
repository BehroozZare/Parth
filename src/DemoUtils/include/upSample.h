//
// Created by behrooz zare on 2024-04-25.
//

#ifndef PARTH_SOLVER_UPSAMPLE_H
#define PARTH_SOLVER_UPSAMPLE_H

#include <Eigen/Core>
#include <vector>


namespace PARTHDEMO {
    Eigen::VectorXi upSampleThePatch(Eigen::VectorXi &I, Eigen::MatrixXi &F,
                          Eigen::MatrixXd &V, Eigen::MatrixXi &F_up,
                          Eigen::MatrixXd &V_up, std::vector<int>& new_to_old_dof_map);
}


#endif //PARTH_SOLVER_DECIMATION_H
