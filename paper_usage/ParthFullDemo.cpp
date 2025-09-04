//
// Created by behrooz zare on 2024-04-06.
//

#include <Eigen/Core>
#include <igl/false_barycentric_subdivision.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/read_triangle_mesh.h>
#include <igl/avg_edge_length.h>
#include <iostream>
#include <cholmod.h>
#include "remesh_botsch.h"
#include <igl/embree/unproject_onto_mesh.h>
#include <igl/vertex_triangle_adjacency.h>
#include <algorithm>
#include <queue>

Eigen::MatrixXd OV, V;
Eigen::MatrixXi OF, F;
//int subdivision_step = 5;
double h;
int step = 1;
int num_decimation_steps = 5;
int num_subdivision_steps = 5;
int num_partial_decimation_steps = 5;
int num_partial_subdivision_steps = 5;

enum SimStage {
    DECIMATION,
    SUBDIVISION,
    PARTIAL_DECIMATION,
    PARTIAL_SUBDIVISION
};



void reset(){
    V = OV;
    F = OF;
    h = igl::avg_edge_length(V,F);
    step = 1;
}


bool subdivision(){

}


bool remesh() {
    Eigen::VectorXd target;
    int n = V.rows();
    target = Eigen::VectorXd::Constant(n,h * step);
    remesh_botsch(V, F);
    step++;




    return true;

}

void UpdateViewer(igl::opengl::glfw::Viewer &viewer) {
    viewer.data().clear();
    viewer.data().set_mesh(V, F);
    viewer.data().set_face_based(true);
    double zoom = 1.0;
    viewer.core().background_color << 1.0f, 1.0f, 1.0f, 0.0f;
    viewer.data().show_lines = true;
    viewer.core().lighting_factor = 1.0;
    viewer.data().show_texture = true;
    viewer.core().orthographic = false;
    viewer.core().camera_zoom *= zoom;
    viewer.core().animation_max_fps = 2.0;
    viewer.data().point_size = 5;
    viewer.data().show_overlay = true;
//    viewer.core().trackball_angle = Eigen::Quaternionf(
//            Eigen::AngleAxisf(M_PI_4 / 2.0, Eigen::Vector3f::UnitX()));
}


int main(int argc, char *argv[]) {

//  igl::read_triangle_mesh("/Users/behroozzare/Desktop/Graphics/ParthSolverDev/data/jello_hi.obj",
//                          OV, OF);
//    igl::read_triangle_mesh("/Users/behroozzare/Desktop/Graphics/ParthSolverDev/data/toy/bar_2d.obj",
//                            OV, OF);
    igl::read_triangle_mesh(argv[1],
                            OV, OF);
    V = OV;
    std::cout << "Number of DOFS: " << V.rows() << std::endl;
    F = OF;
    h = igl::avg_edge_length(V,F);
    igl::opengl::glfw::Viewer viewer;
    SimStage stage = DECIMATION;


    viewer.callback_key_pressed =
            [&](igl::opengl::glfw::Viewer &viewer, unsigned int key, int mods) -> bool {
                switch (key) {
                    default:
                        return false;
                    case ' ':
                        viewer.core().is_animating = !viewer.core().is_animating;
                        if(viewer.core().is_animating){
                            std::cout << "Start animating" << std::endl;
                        } else {
                            std::cout << "Stop animating" << std::endl;
                        }
                        return true;
                    case 'r':
                        std::cout << "Reseting the mesh" << std::endl;
                        reset();
                        UpdateViewer(viewer);
                        step = 1;
                        return true;
                    case 'f':
                        std::cout << "Computing one step of the current stage" << std::endl;
                        if(stage == DECIMATION && step <= num_decimation_steps){
                            remesh();
                        } else {
                            stage = DECIMATION;
                            step = 1;
                        }
                        UpdateViewer(viewer);
                        return true;
                }
                return false;
            };


    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer& viewer) -> bool {
        if (viewer.core().is_animating) {
            if(stage == DECIMATION && step <= num_decimation_steps){
                remesh();
            } else if(stage == SUBDIVISION && step <= num_subdivision_steps){
                std::cerr << "Subdivision is not implemented yet" << std::endl;
                step++;
            } else if(stage == PARTIAL_DECIMATION && step <= num_partial_decimation_steps){
                std::cerr << "Partial Decimation is not implemented yet" << std::endl;
                step++;
            } else if(stage == PARTIAL_SUBDIVISION && step <= num_partial_subdivision_steps){
                std::cerr << "Partial Subdivision is not implemented yet" << std::endl;
                step++;
            }
            if(step % num_decimation_steps == 0){
                stage = SUBDIVISION;
                step = 1;
                reset();
            } else if(step % num_subdivision_steps == 0){
                stage = PARTIAL_DECIMATION;
                step = 1;
                reset();
            } else if(step % num_partial_decimation_steps == 0){
                stage = PARTIAL_SUBDIVISION;
                step = 1;
                reset();
            } else if(step % num_partial_subdivision_steps == 0){
                stage = DECIMATION;
                step = 1;
                reset();
            }
            UpdateViewer(viewer);
        }
        return false;
    };



    viewer.callback_mouse_down = [&](igl::opengl::glfw::Viewer& viewer, int, int)->bool
            {
                int fid;
                Eigen::Vector3f bc;
                // Cast a ray in the view direction starting from the mouse position
                double x = viewer.current_mouse_x;
                double y = viewer.core().viewport(3) - viewer.current_mouse_y;
                igl::embree::EmbreeIntersector ei;
                ei.init(V.cast<float>(),F);
                Eigen::VectorXi Fp;
                Eigen::VectorXi Fi;
                igl::vertex_triangle_adjacency(F, V.rows(), Fi, Fp);
                // Initialize white
                Eigen::MatrixXd C = Eigen::MatrixXd::Constant(F.rows(),3,1);
                if(igl::embree::unproject_onto_mesh(Eigen::Vector2f(x,y), viewer.core().view,
                                                    viewer.core().proj, viewer.core().viewport, ei, fid, bc))
                {
                    // paint hit red
                    std::vector<int> Nfaces;
                    std::queue<int> ring;
                    std::vector<bool> visited(F.rows(), false);
                    ring.push(fid);
                    int total_ring = 4;
                    int ring_cnt = 0;
                    int current_ring_end = 1;
                    int num_pop = 0;
                    while(!ring.empty()){
                        int curr_f = ring.front();
                        ring.pop();
                        num_pop++;
                        //For all faces
                        for(int v_ptr = 0; v_ptr < 3; v_ptr++){
                            int v = F(curr_f,v_ptr);
                            for(int f_ptr = Fp[v]; f_ptr < Fp[v+1]; f_ptr++){
                                int f = Fi(f_ptr);
                                if(!visited[f]){
                                    visited[f] = true;
                                    Nfaces.emplace_back(f);
                                    ring.push(f);
                                }
                            }
                        }
                        if(num_pop == current_ring_end){
                            ring_cnt++;
                            if(ring_cnt == total_ring){
                                break;
                            }
                            current_ring_end = ring.size();
                            num_pop = 0;
                        }
                    }
                    //Compute the adjacency of a face
                    for(auto& f: Nfaces){
                        C.row(f)<<1,0,0;
                    }
                    viewer.data().set_colors(C);
                    return true;
                }
                return false;
            };



    viewer.core().is_animating = true;
    UpdateViewer(viewer);
    viewer.core().is_animating = false;
    viewer.launch();


    return 0;
}