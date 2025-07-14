//
// Created by behrooz on 2022-06-14.
//

#include "Parth_Commander.h"

namespace PARTH {

ParthCommander::ParthCommander(int max_iter_sol_hist)
{
    iter_sol_history.reset(max_iter_sol_hist);
    this->less_than_th_cnt = 0;
}

void ParthCommander::flattenMatrix(Eigen::MatrixXd& V, Eigen::VectorXd& FV)
{
    assert(V.rows() != 0);
    assert(V.cols() != 0);
    FV.resize(V.rows() * V.cols());
    for (int vI = 0; vI < V.rows(); vI++) {
        FV.segment(vI * 3, V.cols()) = V.row(vI).transpose();
    }
}

void ParthCommander::recordIterSol(Eigen::MatrixXd& V)
{
    Eigen::VectorXd q;
    flattenMatrix(V, q);
    recordIterSol(q);
}

void ParthCommander::recordIterSol(Eigen::VectorXd& q)
{
    double dist_norm = (q - prev_sol).norm();
    if (dist_norm < dist_norm_th) {
        less_than_th_cnt++;
    }
    else {
        less_than_th_cnt = 0;
    }
    this->prev_sol = q;
}

SolverType ParthCommander::decideSolver(bool mute) const
{
    if (!mute) {
        std::cout << "The solver is ";
    }
    if (less_than_th_cnt >= 3) { // TODO: This "3" should be more wise
        if (!mute) {
            std::cout << "LBFGS" << std::endl;
        }
        return SolverType::LBFGS_SOLVER;
    }
    if (!mute) {
        std::cout << "Direct" << std::endl;
    }
    return SolverType::DIRECT_SOLVER;
}

void ParthCommander::initNewFrameVars(bool mute)
{
    if (!mute) {
        std::cout << "Resting variables in new frame" << std::endl;
    }
    less_than_th_cnt = 0;
}

}
