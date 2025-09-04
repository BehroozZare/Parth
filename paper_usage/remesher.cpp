//
// Created by behrooz zare on 2024-04-06.
//
#include "remesh_botsch.h"
#include <igl/read_triangle_mesh.h>
#include <igl/write_triangle_mesh.h>

using namespace std;

int main(int argc, const char* argv[]){


    Eigen::MatrixXd V;
    Eigen::VectorXd target;
    Eigen::MatrixXi F;
    string in,out;
    bool project = false;
    int iterations = 10;
    double h = 0.05;
    out = "output.obj";
    int argindex = 0;
//    if(argc>1){
//        in = argv[1];
//        argindex = 1;
//    }else{
//        std::cout<<R"(
//No input specified. To run in command line, issue
//
//./remeshmesh [input.ext] [output.ext] [-i num_iterations] [-h target_edge_length]
//
//The mesh in input.ext must be closed and manifold,
//and ext can be either of obj, ply, off, stl or mesh.
//)";
//        return argindex;
//    }
//    while(argindex+1<argc){
//        if(strncmp(argv[argindex+1],"-i",2)==0){
//            iterations = atoi(argv[argindex+2]);
//            argindex = argindex+2;
//        }else{
//            if(strncmp(argv[argindex+1],"-h",2)==0){
//                h = atof(argv[argindex+2]);
//                argindex = argindex+2;
//            }else{
//                if(strncmp(argv[argindex+1],"-p",2)==0){
//                    project = true;
//                    argindex = argindex+1;
//                }else{
//                    out = argv[argindex+1];
//                    argindex = argindex+1;
//                }}}
//    }
    in = "/Users/behroozzare/Desktop/Graphics/ParthSolverDev/data/libigl-tutorial-data/armadillo.obj";
    out = "/Users/behroozzare/Desktop/Graphics/ParthSolverDev/output/Remesher/armadillo_Remesher.obj";
    iterations = 10;
    h = 0.02;
    project = true;

    igl::read_triangle_mesh(in,V,F);
    std::cout << "V.rows() = " << V.rows() << std::endl;
    std::cout << "F.rows() = " << F.rows() << std::endl;
    std::cout << in << std::endl;
    target = Eigen::VectorXd::Constant(V.rows(),h);
    remesh_botsch(V,F,target,iterations,project);
    igl::write_triangle_mesh(out,V,F);
}
