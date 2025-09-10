#include <iostream>
#include <parth/parth.h>
#include <CLI/CLI.hpp>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <igl/read_triangle_mesh.h>
#include <igl/cotmatrix.h>
#include "utils/polyscope_utils.h"



struct CLIArgs {
    std::string input_mesh;
    std::string output_address;

    CLIArgs(int argc, char *argv[]) {
        CLI::App app{"Separator analysis"};

        app.add_option("-o,--output", output_address, "output folder name");
        app.add_option("-i,--input", input_mesh, "input mesh name");

        try {
            app.parse(argc, argv);
        } catch (const CLI::ParseError &e) {
            exit(app.exit(e));
        }
    }
};



int main(int argc, char *argv[]) {
    // Load the mesh
    CLIArgs args(argc, argv);

    if (args.input_mesh.empty()) {
        std::cerr << "Error: Input mesh file not specified. Use -i or --input to specify the mesh file." << std::endl;
        return 1;
    }

    std::cout << "Loading mesh from: " << args.input_mesh << std::endl;
    std::cout << "Output folder: " << args.output_address << std::endl;

    Eigen::MatrixXd OV;
    Eigen::MatrixXi OF;
    if (!igl::read_triangle_mesh(args.input_mesh, OV, OF)) {
        std::cerr << "Failed to read the mesh: " << args.input_mesh << std::endl;
        return 1;
    }

    //Create laplacian matrix
    Eigen::SparseMatrix<double> OL;
    igl::cotmatrix(OV, OF, OL);

    //init Parth
    std::vector<int> perm;
    PARTH::ParthAPI parth;
    parth.setMatrix(OL.rows(), OL.outerIndexPtr(), OL.innerIndexPtr(), 1);
    parth.computePermutation(perm);

    //Group nodes into patch and non-patch
    std::vector<bool> is_separator(OL.rows(), false);
    for (int region = 0; region < parth.hmd.HMD_tree.size(); region++){
        auto& node = parth.hmd.HMD_tree[region];
        if (!node.isLeaf()){
            auto& dofs = node.DOFs;
            for (int dof : dofs){
                is_separator[dof] = true;
            }
        }
    }

    // Initialize polyscope
    PARTH::utils::polyscope_utils::initialize();
    
    // Create vertex colors based on separator status
    // Grey for normal vertices (0.5, 0.5, 0.5), Black for separator vertices (0.0, 0.0, 0.0)
    Eigen::MatrixXd vertex_colors(OV.rows(), 3);
    for (int i = 0; i < OV.rows(); i++) {
        if (is_separator[i]) {
            // Black for separator vertices
            vertex_colors.row(i) << 0.0, 0.0, 0.0;
        } else {
            // Grey for normal vertices
            vertex_colors.row(i) << 0.5, 0.5, 0.5;
        }
    }
    
    // Register the mesh with polyscope
    PARTH::utils::polyscope_utils::registerSurfaceMesh("input_mesh", OV, OF);
    
    // Add vertex colors to visualize separator vs normal vertices
    PARTH::utils::polyscope_utils::addVertexColors("input_mesh", "separator_colors", vertex_colors);
    
    std::cout << "\nVisualization Info:" << std::endl;
    std::cout << "- Black vertices: Separator nodes" << std::endl;
    std::cout << "- Grey vertices: Normal nodes" << std::endl;
    std::cout << "Total vertices: " << OV.rows() << std::endl;
    
    int separator_count = 0;
    for (bool sep : is_separator) {
        if (sep) separator_count++;
    }
    std::cout << "Separator vertices: " << separator_count << std::endl;
    std::cout << "Normal vertices: " << (OV.rows() - separator_count) << std::endl;
    
    // Show the polyscope GUI
    PARTH::utils::polyscope_utils::show();
    
    return 0;
}