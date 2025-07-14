//
// Created by behrooz zare on 2024-04-25.
//

#include "remeshPatch.h"

#include "igl/slice.h"
#include "igl/upsample.h"
#include "igl/boundary_facets.h"
#include "igl/unique.h"
#include "igl/setdiff.h"
#include "igl/colon.h"
#include "igl/avg_edge_length.h"
#include "remesh_botsch.h"

namespace PARTHDEMO {

    Eigen::VectorXi remesh(Eigen::VectorXi &I, Eigen::MatrixXi &F,
                           Eigen::MatrixXd &V, Eigen::MatrixXi &F_dec,
                           Eigen::MatrixXd &V_dec, double scale, std::vector<int>& new_to_old_dof_map) {
        // Compute a local face set and vertices set
        Eigen::MatrixXi F_sub;
        Eigen::MatrixXd V_sub;

        // Assign the local face set and vertices set
        igl::slice(F, I, 1, F_sub);
        std::vector<int> local_to_global_vertices;
        std::vector<int> global_to_local_vertices;
        // Compute the set of vertices used in the F_sub by iterating over each face
        for (int r = 0; r < F_sub.rows(); r++) {
            // Add each vertex of the face to the set of vertices
            for (int c = 0; c < F_sub.cols(); c++) {
                local_to_global_vertices.push_back(F_sub(r, c));
            }
        }

        // Delete duplicated vertices and sort the vertices_ids
        std::sort(local_to_global_vertices.begin(), local_to_global_vertices.end());
        local_to_global_vertices.erase(
                std::unique(local_to_global_vertices.begin(), local_to_global_vertices.end()),
                local_to_global_vertices.end());

        // Create V_sub using vertices_ids
        V_sub.resize(local_to_global_vertices.size(), V.cols());
        for (int i = 0; i < local_to_global_vertices.size(); i++) {
            V_sub.row(i) = V.row(local_to_global_vertices[i]);
        }

        // Compute global to local ids
        global_to_local_vertices.resize(V.rows(), -1);
        for (int i = 0; i < local_to_global_vertices.size(); i++) {
            global_to_local_vertices[local_to_global_vertices[i]] = i;
        }

        // Map the F_sub based on local vertices
        for (int r = 0; r < F_sub.rows(); r++) {
            for (int c = 0; c < F_sub.cols(); c++) {
                assert(global_to_local_vertices[F_sub(r, c)] != -1);
                F_sub(r, c) = global_to_local_vertices[F_sub(r, c)];
            }
        }



        // ------------------- Decimate the selected faces -------------------
        int prev_num_nodes = V_sub.rows();
        int prev_num_faces = F_sub.rows();

//    igl::decimate(V_sub, F_sub, num_faces, V_sub_new, F_sub_new, F_sub_new_idx, V_sub_new_idx);
        // Find boundary edges
        Eigen::MatrixXi E;
        igl::boundary_facets(F_sub, E);
        // Find boundary vertices
        Eigen::VectorXi b, IA, IC;
        igl::unique(E, b, IA, IC);

        // List of all vertex indices
        Eigen::VectorXi all, in;
        igl::colon<int>(0, V_sub.rows() - 1, all);
        // List of interior indices
        igl::setdiff(all, b, in, IA);
        //Concatenate b and in
        Eigen::VectorXi order(b.rows() + in.rows());
        Eigen::VectorXi new_label(order.rows());
        assert(order.rows() == V_sub.rows());
        order << b, in;

        for (int i = 0; i < new_label.rows(); i++) {
            new_label(order(i)) = i;
        }

        //Rename faces
        for (int i = 0; i < F_sub.rows(); i++) {
            for (int j = 0; j < F_sub.cols(); j++) {
                F_sub(i, j) = new_label(F_sub(i, j));
            }
        }

        Eigen::MatrixXd V_sub_reordered(V_sub.rows(), V_sub.cols());
        for (int i = 0; i < order.rows(); i++) {
            V_sub_reordered.row(i) = V_sub.row(order(i));
        }
        V_sub = V_sub_reordered;
        for (int i = 0; i < b.rows(); i++) {
            b(i) = i;
        }

        //Compute the correct local to global mapping by assuming that the boundary vertices
        //are only exist and all the other vertices are deleted.
        std::vector<int> tmp(V_sub.rows());
        for (int i = 0; i < V_sub.rows(); i++) {
            int prev_local_id = order[i];
            tmp[i] = local_to_global_vertices[prev_local_id];
        }
        local_to_global_vertices = tmp; // Updating the local to global with v_dec

        double h = igl::avg_edge_length(V, F);
        assert(b.rows() != V_sub.rows());
        std::cout << "Number of faces " << F_sub.rows() << std::endl;
        std::cout << "Number of vertices " << V_sub.rows() << std::endl;
        Eigen::VectorXd target;
        target = Eigen::VectorXd::Constant(V_sub.rows(),h * scale);
        std::vector<int> patch_old_to_new_dof_map;
        remesh_botsch_map(V_sub, F_sub, patch_old_to_new_dof_map, target, 10, b, false);
        std::cout <<" -------- After Remesh --------" << std::endl;
        std::cout << "Number of faces " << F_sub.rows() << std::endl;
        std::cout << "Number of vertices " << V_sub.rows() << std::endl;



#ifndef NDEBUG
        // Find boundary edges
        igl::boundary_facets(F_sub, E);
        // Find boundary vertices
        Eigen::VectorXi b_new;
        igl::unique(E, b_new, IA, IC);
        assert(b_new.rows() == b.rows());
        for (int i = 0; i < b.rows(); i++) {
            assert(b_new(i) == i);
            patch_old_to_new_dof_map[i] = i;
        }
#endif


        // ------------------- Integrate the up decimated patch back to the full mesh

        // ------ Integrate vertices
        //Create a new local to global vertices mapping that integrated with deleted nodes
        std::vector<bool> vertex_is_deleted(V.rows(), false);
        for (int i = 0; i < prev_num_nodes; i++) {
            if (i < b.rows()) {
                vertex_is_deleted[local_to_global_vertices[i]] = false;
            } else {
                vertex_is_deleted[local_to_global_vertices[i]] = true;
            }
        }
        assert(local_to_global_vertices.size() == order.rows());


        int v_cnt = 0;

        new_to_old_dof_map.clear();
        V_dec.resize(V.rows() - prev_num_nodes + V_sub.rows(), V.cols());
        for (int v = 0; v < V.rows(); v++) {
            if (!vertex_is_deleted[v]) {
                new_to_old_dof_map.emplace_back(v);
                V_dec.row(v_cnt++) = V.row(v);
            }
        }

        int num_constant_vertices = v_cnt;

        for (int v = b.rows(); v < V_sub.rows(); v++) {
            new_to_old_dof_map.emplace_back(-1);
            V_dec.row(v_cnt++) = V_sub.row(v);
        }
        assert(v_cnt == V_dec.rows());

        // ------ Integrate faces
        //Compute old to new dof mapping
        std::vector<int> old_to_new_dof_map(V.rows(), -1);
        for (int n = 0; n < V_dec.rows(); n++) {
            if (new_to_old_dof_map[n] != -1) {
                assert(new_to_old_dof_map[n] < V.rows());
                old_to_new_dof_map[new_to_old_dof_map[n]] = n;
            }
        }

        std::vector<bool> face_is_chosen(F.rows(), false);
        for (int i = 0; i < I.rows(); i++) {
            face_is_chosen[I(i)] = true;
        }

        int total_faces = F.rows() - prev_num_faces + F_sub.rows();
        F_dec.resize(total_faces, F.cols());
        std::vector<int> patch_indices;
        int f_cnt = 0;
        for (int f = 0; f < F.rows(); f++) {
            if (!face_is_chosen[f]) {
                for (int c = 0; c < F.cols(); c++) {
                    assert(old_to_new_dof_map[F.row(f)(c)] != -1 && old_to_new_dof_map[F.row(f)(c)] < V_dec.rows());
                    F_dec.row(f_cnt)(c) = old_to_new_dof_map[F.row(f)(c)];
                }
                f_cnt++;
            }
        }


        for (int f = 0; f < F_sub.rows(); f++) {
            for (int c = 0; c < F_sub.cols(); c++) {
                int v_local = F_sub.row(f)(c);
                if(v_local < b.rows()){
                    int v_global_old = local_to_global_vertices[v_local];
                    int v_global_new = old_to_new_dof_map[v_global_old];
                    assert(v_global_new != -1 && v_global_new < V_dec.rows());
                    F_dec.row(f_cnt)(c) = v_global_new;
                } else {
                    int v_global_new = num_constant_vertices + v_local - b.rows();
                    assert(v_global_new != -1 && v_global_new < V_dec.rows());
                    F_dec.row(f_cnt)(c) = v_global_new;
                }
            }
            patch_indices.emplace_back(f_cnt);
            f_cnt++;
        }
        assert(total_faces == f_cnt);

        std::cout << "------------- coloring faces: " << patch_indices.size() << std::endl;

        Eigen::VectorXi patch(patch_indices.size());
        for (int i = 0; i < patch.rows(); i++) {
            patch(i) = patch_indices[i];
        }
        return patch;
    }

}