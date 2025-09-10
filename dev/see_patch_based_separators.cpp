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

    CLIArgs(int argc, char *argv[]) {
        CLI::App app{"Separator analysis and visualization"};

        app.add_option("-i,--input", input_mesh, "input mesh name")->required();

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

    //Create sub-mesh from separators
    std::vector<int> chosen_graph_nodes(is_separator.size(), 0);
    for (int i = 0; i < OV.rows(); i++) {
        if (is_separator[i]) {
            chosen_graph_nodes[i] = 1;
        }
    }
    PARTH::ParthAPI sub_parth;
    PARTH::SubMesh sub_mesh;
    assert(parth.getLagLevel() > 0);
    sub_parth.setNDLevels(parth.getLagLevel() - 1);
    parth.hmd.createSubMesh(chosen_graph_nodes, parth.M_n, parth.Mp, parth.Mi, sub_mesh);
    sub_parth.setMesh(sub_mesh.M_n, sub_mesh.Mp.data(), sub_mesh.Mi.data());
    sub_parth.computePermutation(perm);
    std::vector<int> region_level(sub_parth.hmd.HMD_tree.size(), 0);
    for (int region = 0; region < sub_parth.hmd.HMD_tree.size(); region++){
        auto& node = sub_parth.hmd.HMD_tree[region];
        region_level[region] = node.level;
    }

    std::vector<int> global_to_local_DOF_id(parth.M_n, -1);
    for (int i = 0; i < sub_mesh.local_to_global_DOF_id.size(); i++){
        global_to_local_DOF_id[sub_mesh.local_to_global_DOF_id[i]] = i;
    }

    // Initialize polyscope
    PARTH::utils::polyscope_utils::initialize();
    
    // Register the mesh with polyscope
    PARTH::utils::polyscope_utils::registerSurfaceMesh("input_mesh", OV, OF);
    
    // Create multiple vertex color quantities for cumulative levels
    int max_level = parth.hmd.num_levels;
    
    // Create color quantities for each cumulative level (1 to max_level)
    for (int target_level = 0; target_level <= max_level; target_level++) {
        Eigen::MatrixXd vertex_colors(OV.rows(), 3);
        
        for (int i = 0; i < OV.rows(); i++) {
            if (is_separator[i]) {
                // Get the level of this separator vertex
                int local_node = global_to_local_DOF_id[i];
                assert(local_node != -1);
                int vertex_level = region_level[sub_parth.hmd.DOF_to_HMD_node[local_node]];
                
                // Only color separators up to the target level
                if (vertex_level <= target_level) {
                    // Normalize level to [0, 1] range based on max_level
                    double normalized_level = max_level > 0 ? static_cast<double>(vertex_level) / static_cast<double>(max_level) : 0.0;
                    
                    // Create a color gradient from blue (level 0) to red (max level)
                    double red = normalized_level;           // 0 at level 0, 1 at max level
                    double green = 0.0;                      // Keep green low for better contrast
                    double blue = 1.0 - normalized_level;    // 1 at level 0, 0 at max level
                    
                    vertex_colors.row(i) << red, green, blue;
                } else {
                    // Grey for separators above target level (not colored)
                    vertex_colors.row(i) << 0.5, 0.5, 0.5;
                }
            } else {
                // Grey for normal vertices
                vertex_colors.row(i) << 0.5, 0.5, 0.5;
            }
        }
        
        // Add this color quantity with a descriptive name
        std::string quantity_name = "Level_" + std::to_string(target_level) + "_separators";
        PARTH::utils::polyscope_utils::addVertexColors("input_mesh", quantity_name, vertex_colors);
    }
    
    // Also create an "All Levels" quantity that shows all separators
    Eigen::MatrixXd all_vertex_colors(OV.rows(), 3);
    for (int i = 0; i < OV.rows(); i++) {
        if (is_separator[i]) {
            // Color based on level for separator vertices
            int level = region_level[parth.hmd.DOF_to_HMD_node[i]];
            
            // Normalize level to [0, 1] range
            double normalized_level = max_level > 0 ? static_cast<double>(level) / static_cast<double>(max_level) : 0.0;
            
            // Create a color gradient from blue (level 0) to red (max level)
            double red = normalized_level;           // 0 at level 0, 1 at max level
            double green = 0.0;                      // Keep green low for better contrast
            double blue = 1.0 - normalized_level;    // 1 at level 0, 0 at max level
            
            all_vertex_colors.row(i) << red, green, blue;
        } else {
            // Grey for normal vertices
            all_vertex_colors.row(i) << 0.5, 0.5, 0.5;
        }
    }
    PARTH::utils::polyscope_utils::addVertexColors("input_mesh", "All_levels_separators", all_vertex_colors);
    
    std::cout << "\nVisualization Info:" << std::endl;
    std::cout << "- Created " << max_level << " cumulative level quantities: Level_1_separators to Level_" << max_level << "_separators" << std::endl;
    std::cout << "- Plus 'All_levels_separators' showing all separator levels" << std::endl;
    std::cout << "- Blue to Red gradient: Separator nodes (Blue=level 0, Red=max level " << max_level << ")" << std::endl;
    std::cout << "- Grey vertices: Normal nodes and higher-level separators (when viewing cumulative quantities)" << std::endl;
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