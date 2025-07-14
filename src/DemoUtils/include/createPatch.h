//
// Created by behrooz zare on 2024-04-25.
//

#ifndef PARTH_SOLVER_RANDOMPATCH_H
#define PARTH_SOLVER_RANDOMPATCH_H

#include <Eigen/Core>
#include <igl/vertex_triangle_adjacency.h>
#include <vector>
#include <queue>
#include <algorithm>


namespace PARTHDEMO {
    void createPatch(int fid, ///<[in] center of the patch
                     double ring_size, ///<[in] how many ring of neighbors in BFS around a face should be included
                     Eigen::VectorXi &SelectedFaces,///<[in] selected faces ids
                     Eigen::MatrixXi &F,///<[in] Faces
                     Eigen::MatrixXd &V///<[in] Vertices
                     );
}


#endif //PARTH_SOLVER_DECIMATION_H
