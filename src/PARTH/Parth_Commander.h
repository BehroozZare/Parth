//
// Created by behrooz on 2022-06-14.
//

#ifndef PARTH_COMMANDER_H
#define PARTH_COMMANDER_H

#include <vector>
#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <iostream>

#include "csv_utils.h"
#include "Parth_utils.h"
#include "RingBuffer.h"

namespace PARTH {

enum SolverType { DIRECT_SOLVER,
    LBFGS_SOLVER };

///---------------------------------------------------------------------------------------\n
/// ParthCommander - A class that decide the combination of the solution based on recorded history\n
///---------------------------------------------------------------------------------------\n
class ParthCommander {
protected:
    ///---------------------------------------------------------------------------------------\n
    /// flattenMatrix - Flatten the R * C matrix into a vector by row major order\n
    ///---------------------------------------------------------------------------------------\n
    static void flattenMatrix(Eigen::MatrixXd& V, ///<[in] Input matrix
        Eigen::VectorXd& FV ///[out] output flatten vector
    );

private:
    // Recorded features storage
    RingStorage<Eigen::VectorXd> iter_sol_history; ///< @brief The history of the newton iterations of the current frame

    // The dist norm feature variables
    std::vector<double> dist_norm; ///< @brief store the distance norm of two consecutive iterations solution
    std::vector<double> dist_norm_derivative; /// < @brief store the derivative of the distance norm

    double dist_norm_th = 0.01; /// <@brief if the distance is lass than this threshold it assumed to be small
    Eigen::VectorXd prev_sol; /// <@brief TODO: delete after finishing ring storage>
    int less_than_th_cnt; ///<@brief TODO: delete after finishing ring storage>

public:
    ParthCommander(int max_iter_sol_hist = 5);
    ~ParthCommander() = default;

    ///---------------------------------------------------------------------------------------\n
    /// recordIterSol - Record the solution of a single iteration\n
    ///---------------------------------------------------------------------------------------\n
    void recordIterSol(Eigen::MatrixXd& V ///<[in] The position of the elements in a R * 3 style
    );

    ///---------------------------------------------------------------------------------------\n
    /// recordIterSol - Record the solution of a single iteration\n
    ///---------------------------------------------------------------------------------------\n
    void recordIterSol(Eigen::VectorXd& q ///<[in] The position of the elements in a vector [x y z] per elements
    );

    void recordDistance(double distance)
    { /// TODO: Delete this function after integration of the frameworks
        if (distance < dist_norm_th) {
            less_than_th_cnt++;
        }
        else {
            less_than_th_cnt = 0;
        }
    }

    ///---------------------------------------------------------------------------------------\n
    /// decideSolver - Decide a solver based on recorded features (Currently supports LBFGS and Direct Solver)\n
    ///---------------------------------------------------------------------------------------\n
    SolverType decideSolver(bool mute = false) const;

    ///---------------------------------------------------------------------------------------\n
    /// initNewFrameVars - reset variables that need to be reset in each frame
    ///---------------------------------------------------------------------------------------\n
    void initNewFrameVars(bool mute = false);
};
}
#endif // IPC_PARTH_COMMANDER_H
