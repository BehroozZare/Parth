//
// Created by behrooz zare on 2024-04-25.
//

#include "igl/slice.h"
#include "igl/upsample.h"

namespace PARTHDEMO {
    Eigen::VectorXi upSampleThePatch(Eigen::VectorXi &I, Eigen::MatrixXi &F,
                          Eigen::MatrixXd &V, Eigen::MatrixXi &F_up,
                          Eigen::MatrixXd &V_up, std::vector<int>& new_to_old_dof_map) {

        std::vector<bool> index_chosen(F.rows(), false);
        for (int i = 0; i < I.rows(); i++) {
            index_chosen[I(i)] = true;
        }

        // Compute a local face set and vertices set
        Eigen::MatrixXi F_sub;
        Eigen::MatrixXd V_sub;

        // Assign the local face set and vertices set
        igl::slice(F, I, 1, F_sub);
        std::vector<int> local_to_global_map;
        std::vector<int> global_to_local_map;
        // Compute the set of vertices used in the F_sub by iterating over each face
        for (int r = 0; r < F_sub.rows(); r++) {
            // Add each vertex of the face to the set of vertices
            for (int c = 0; c < F_sub.cols(); c++) {
                local_to_global_map.push_back(F_sub(r, c));
            }
        }

        // Delete duplicated vertices and sort the vertices_ids
        std::sort(local_to_global_map.begin(), local_to_global_map.end());
        local_to_global_map.erase(
                std::unique(local_to_global_map.begin(), local_to_global_map.end()),
                local_to_global_map.end());

        // Create V_sub using vertices_ids
        V_sub.resize(local_to_global_map.size(), V.cols());
        for (int i = 0; i < local_to_global_map.size(); i++) {
            V_sub.row(i) = V.row(local_to_global_map[i]);
        }

        // Compute global to local ids
        global_to_local_map.resize(V.rows());
        for (int i = 0; i < local_to_global_map.size(); i++) {
            global_to_local_map[local_to_global_map[i]] = i;
        }

        // Map the F_sub based on local vertices
        for (int r = 0; r < F_sub.rows(); r++) {
            for (int c = 0; c < F_sub.cols(); c++) {
                F_sub(r, c) = global_to_local_map[F_sub(r, c)];
            }
        }

        // ------------------- Up sample the selected faces -------------------
        int prev_num_nodes = V_sub.rows();
        int prev_num_faces = F_sub.rows();
        igl::upsample(Eigen::MatrixXd(V_sub), Eigen::MatrixXi(F_sub), V_sub, F_sub);

        // ------------------- Integrate the up sampled Vertices and faces
        // ------ Integrate vertices
        V_up.resize(V.rows() + V_sub.rows() - prev_num_nodes, V.cols());
        for (int i = 0; i < V.rows(); i++) {
            V_up.row(i) = V.row(i);
        }

        for (int i = prev_num_nodes; i < V_sub.rows(); i++) {
            V_up.row(V.rows() + i - prev_num_nodes) = V_sub.row(i);
        }

        // ------ Integrate faces

        // node mapping
        local_to_global_map.resize(V_sub.rows());
        for (int i = 0; i < V_sub.rows() - prev_num_nodes; i++) {
            local_to_global_map[i + prev_num_nodes] = V.rows() + i;
            assert(local_to_global_map[i] < V_up.rows() && "Index out of bound");
        }

        // Map the faces to their global vertices
        F_up.resize(F.rows() + F_sub.rows() - prev_num_faces, F.cols());
        int cnt = 0;
        for (int i = 0; i < F.rows(); i++) {
            if (!index_chosen[i]) {
                F_up.row(cnt++) = F.row(i);
            }
        }

        std::vector<int> patch_indices;
        for (int j = 0; j < F_sub.rows(); j++) {
            for (int i = 0; i < F_sub.cols(); i++) {
                F_up.row(cnt)(i) = local_to_global_map[F_sub.row(j)(i)];
                if (F_up.row(cnt)(i) >= V_up.rows()) {
                    std::cout << F_up.row(cnt)(i) << " >= " << V_up.rows()
                              << " - Index out of bound" << std::endl;
                }
                assert(F_up.row(cnt)(i) < V_up.rows() && "Index out of bound");
            }
            patch_indices.emplace_back(cnt);
            cnt++;
        }



        new_to_old_dof_map.clear();
        for (int i = 0; i < V.rows(); i++) {
            new_to_old_dof_map.push_back(i);
        }
        for (int j = V.rows(); j < V_up.rows(); j++) {
            new_to_old_dof_map.push_back(-1);
        }

        Eigen::VectorXi patch(patch_indices.size());
        for (int i = 0; i < patch.rows(); i++) {
            patch(i) = patch_indices[i];
        }
        return patch;
    }

}