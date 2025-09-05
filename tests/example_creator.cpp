#include <iostream>
#include <fstream>
#include <CLI/CLI.hpp>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include <igl/read_triangle_mesh.h>
#include <igl/cotmatrix.h>
#include <parth/parth.h>


struct CLIArgs {
    std::string input_mesh;
    std::string output_address;

    CLIArgs(int argc, char *argv[]) {
        CLI::App app{"Example creator"};

        app.add_option("-o,--output", output_address, "output folder name");
        app.add_option("-i,--input", input_mesh, "input mesh name");

        try {
            app.parse(argc, argv);
        } catch (const CLI::ParseError &e) {
            exit(app.exit(e));
        }
    }
};


//Add edges in specific places
void addEdges(Eigen::SparseMatrix<double> &L, std::vector<std::pair<int, int>> &edges) {
    for (int i = 0; i < edges.size(); i++) {
        L.coeffRef(edges[i].first, edges[i].second) = 1;
        L.coeffRef(edges[i].second, edges[i].first) = 1;
    }
}

void convertHmdEdgesToMatrixEdges(std::vector<std::pair<int, int>> &hmd_edges,
    std::vector<std::pair<int, int>> &matrix_edges,
    PARTH::Parth &parth, int num_edges) {
        
    for (int i = 0; i < hmd_edges.size(); i++) {
        std::vector<int>& first_region_dofs = parth.hmd.HMD_tree[hmd_edges[i].first].DOFs;
        std::vector<int>& second_region_dofs = parth.hmd.HMD_tree[hmd_edges[i].second].DOFs;
        if(first_region_dofs.size() == 0 || second_region_dofs.size() == 0) {
            continue;
        }
        
        for(int k = 0; k < num_edges; k++) {
            //Peak a random edges from first and second region
            int random_node_from_first_region = first_region_dofs[rand() % first_region_dofs.size()];
            int random_node_from_second_region = second_region_dofs[rand() % second_region_dofs.size()];
            if(random_node_from_first_region != random_node_from_second_region) {
                matrix_edges.push_back(std::pair<int, int>(random_node_from_first_region,
                        random_node_from_second_region));
                matrix_edges.push_back(std::pair<int, int>(random_node_from_second_region,
                        random_node_from_first_region));
            }
        }
    }
}




int main(int argc, char *argv[]) {
    // Load the mesh
    CLIArgs args(argc, argv);
    
    if (args.input_mesh.empty()) {
        std::cerr << "Error: Input mesh file not specified. Use -i or --input to specify the mesh file." << std::endl;
        return 1;
    }
    
    if (args.output_address.empty()) {
        std::cerr << "Error: Output folder not specified. Use -o or --output to specify the output folder." << std::endl;
        return 1;
    }
    
    std::cout << "Loading mesh from: " << args.input_mesh << std::endl;
    std::cout << "Output folder: " << args.output_address << std::endl;
    
    Eigen::MatrixXd OV, V;
    Eigen::MatrixXi OF, F;
    if (!igl::read_triangle_mesh(args.input_mesh, OV, OF)) {
        std::cerr << "Failed to read the mesh: " << args.input_mesh << std::endl;
        return 1;
    }

    V = OV;
    F = OF;
    std::vector<std::vector<std::pair<int, int>>> test_hmd_edges;
    //Test case 0 -> full reuse
    test_hmd_edges.push_back(std::vector<std::pair<int, int>>({std::pair<int, int>(0, 1),
        std::pair<int, int>(0, 5), std::pair<int, int>(4, 9), std::pair<int, int>(4, 10)}));
    //Test case 1 -> level 1 reuse
    test_hmd_edges.push_back(std::vector<std::pair<int, int>>({std::pair<int, int>(2, 3)}));
    //Test case 2 -> level 2 reuse
    test_hmd_edges.push_back(std::vector<std::pair<int, int>>({std::pair<int, int>(3, 7)}));
    //Test case 3 -> level 3 reuse
    test_hmd_edges.push_back(std::vector<std::pair<int, int>>({std::pair<int, int>(5, 11)}));
    //Test case 4 -> No reuse
    test_hmd_edges.push_back(std::vector<std::pair<int, int>>({std::pair<int, int>(15, 19)}));

    //Create laplacian matrix
    Eigen::SparseMatrix<double> OL, L;
    igl::cotmatrix(V, F, OL);
    //save original matrix
    std::string original_matrix_address = args.output_address + "/original_matrix.mtx";
    Eigen::saveMarket(OL, original_matrix_address);

    for (int test_num = 0; test_num < test_hmd_edges.size(); test_num++) {
        std::vector<std::pair<int, int>> hmd_edges = test_hmd_edges[test_num];
        std::vector<std::pair<int, int>> matrix_edges;
        std::vector<int> perm;
        int num_edges = 3;

        //init Parth
        PARTH::Parth parth;
        parth.setMatrix(OL.rows(), OL.outerIndexPtr(), OL.innerIndexPtr(), 1);
        parth.computePermutation(perm);
        //Add edges
        convertHmdEdgesToMatrixEdges(hmd_edges, matrix_edges, parth, num_edges);
        L = OL;
        addEdges(L, matrix_edges);

        //Update Parth
        parth.setMatrix(L.rows(), L.outerIndexPtr(), L.innerIndexPtr(), 1);
        parth.computePermutation(perm);
        parth.printTiming();
        std::string matrix_address = args.output_address + "/" + std::to_string(test_num) + "_matrix.mtx";
        Eigen::saveMarket(L, matrix_address);
        std::cout << "Number of added edges: " << matrix_edges.size() << std::endl;
        std::cout << "difference between original and full reuse matrix: " << L.nonZeros() - OL.nonZeros() << std::endl;
        std::cout << "\n\n\n" << std::endl;
    }
    
    return 0;
}