#include <iostream>
#include <parth/parth.h>
#include <csv_utils.h>
#include <get_factor_nnz.h>
#include <omp.h>
#include <CLI/CLI.hpp>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <igl/read_triangle_mesh.h>
#include <igl/cotmatrix.h>


struct CLIArgs {
    std::string input_mesh;
    std::string output_address;
    std::string permutation_type = "METIS";

    CLIArgs(int argc, char *argv[]) {
        CLI::App app{"Separator analysis"};

        app.add_option("-o,--output", output_address, "output folder name");
        app.add_option("-i,--input", input_mesh, "input mesh name");
        app.add_option("-p,--permutation", permutation_type, "permutation type");

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

    //Get the mesh name by finding the first "/" from right
    std::string mesh_name = args.input_mesh.substr(args.input_mesh.find_last_of("/") + 1);
    mesh_name = mesh_name.substr(0, mesh_name.find_last_of("."));
    std::cout << "Mesh name: " << mesh_name << std::endl;
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
    if (args.permutation_type == "METIS"){
        parth.setReorderingType(PARTH::ReorderingType::METIS);
    } else if (args.permutation_type == "AMD"){
        parth.setReorderingType(PARTH::ReorderingType::AMD);
    } else {
        std::cerr << "Error: Invalid permutation type."
                     " Use -p or --permutation to specify the permutation type (either METIS or AMD)." << std::endl;
        return 1;
    }

    parth.setMatrix(OL.rows(), OL.outerIndexPtr(), OL.innerIndexPtr(), 1);
    double start_time = omp_get_wtime();
    parth.computePermutation(perm);
    double end_time = omp_get_wtime();

    //Max runtime per-level
    std::vector<double> max_permutation_runtime(parth.getNDLevels(), 0);
    std::vector<double> max_sep_runtime(parth.getNDLevels(), 0);
    double total_sep_time = 0;
    double total_permute_time = 0;
    for (auto & node : parth.hmd.HMD_tree){
        max_permutation_runtime[node.level] = std::max(max_permutation_runtime[node.level], node.permute_time);
        max_sep_runtime[node.level] = std::max(max_sep_runtime[node.level], node.separator_comp_time);
        total_sep_time += node.separator_comp_time;
        total_permute_time += node.permute_time;
    }
    double total_critical_separator_time = 0;
    double total_critical_permute_time = 0;
    //Sum over vectors
    for (int i = 0; i < max_permutation_runtime.size(); i++) {
        total_critical_permute_time += max_permutation_runtime[i];
        total_critical_separator_time += max_sep_runtime[i];
    }

    int A_NNZ = OL.nonZeros();
    Eigen::SparseMatrix<double> OL_lower = OL.triangularView<Eigen::Lower>();
    int L_NNZ = get_factor_nnz(OL_lower.outerIndexPtr(), OL_lower.innerIndexPtr(), OL_lower.valuePtr(),
        OL_lower.rows(), OL_lower.nonZeros(), perm);

    double total_time = end_time - start_time;
    std::cout << "Total time: " << total_time << std::endl;
    std::cout << "Total separator time: " << total_sep_time << std::endl;
    std::cout << "Total permutation time: " << total_permute_time << std::endl;
    std::cout << "Total critical separator time: " << total_critical_separator_time << std::endl;
    std::cout << "Total critical permutation time: " << total_critical_permute_time << std::endl;
    double percentage_of_sep_time = total_sep_time / total_time * 100;
    double percentage_of_permute_time = total_permute_time / total_time * 100;
    double ratio_of_sep_to_permute_time = total_sep_time / total_permute_time;
    std::cout << "Percentage of separator time: " << percentage_of_sep_time << std::endl;
    std::cout << "Percentage of permutation time: " << percentage_of_permute_time << std::endl;
    std::cout << "Ratio of separator to permutation time: " << ratio_of_sep_to_permute_time << std::endl;
    double percentage_of_critical_sep_time = total_critical_separator_time / total_time * 100;
    double percentage_of_critical_permute_time = total_critical_permute_time / total_time * 100;
    double ratio_of_critical_sep_to_critical_permute_time = total_critical_separator_time / total_critical_permute_time;
    std::cout << "Ratio of critical separator to critical permutation time: " << ratio_of_critical_sep_to_critical_permute_time << std::endl;
    std::cout << "Percentage of critical sep time: " << percentage_of_critical_sep_time << std::endl;
    std::cout << "Percentage of critical permute time: " << percentage_of_critical_permute_time << std::endl;
    std::cout << "A_NNZ: " << A_NNZ << std::endl;
    std::cout << "L_NNZ: " << L_NNZ << std::endl;
    std::cout << "L_NNZ/A_NNZ: " << L_NNZ / A_NNZ << std::endl;

    //Write to csv
    std::string csv_name = args.output_address + "/sep_runtime_analysis";
    std::vector<std::string> header;
    header.emplace_back("mesh_name");
    header.emplace_back("N");
    header.emplace_back("A_NNZ");
    header.emplace_back("L_NNZ");
    header.emplace_back("permute_type");
    header.emplace_back("L_NNZ/A_NNZ");
    header.emplace_back("total_time");
    header.emplace_back("total_sep_time");
    header.emplace_back("total_permute_time");
    header.emplace_back("total_critical_sep_time");
    header.emplace_back("total_critical_permute_time");
    header.emplace_back("sep_time/permute_time");
    header.emplace_back("critical_sep_time/critical_permute_time");
    header.emplace_back("percentage_of_sep_time");
    header.emplace_back("percentage_of_permute_time");
    header.emplace_back("percentage_of_critical_sep_time");
    header.emplace_back("percentage_of_critical_permute_time");


    PARTH::CSVManager runtime_csv(csv_name, "some address", header,
                                  false);
    runtime_csv.addElementToRecord(mesh_name, "mesh_name");
    runtime_csv.addElementToRecord(OL.rows(), "N");
    runtime_csv.addElementToRecord(A_NNZ, "A_NNZ");
    runtime_csv.addElementToRecord(L_NNZ, "L_NNZ");
    runtime_csv.addElementToRecord(args.permutation_type, "permute_type");
    runtime_csv.addElementToRecord(L_NNZ / A_NNZ, "L_NNZ/A_NNZ");
    runtime_csv.addElementToRecord(total_time, "total_time");
    runtime_csv.addElementToRecord(total_sep_time, "total_sep_time");
    runtime_csv.addElementToRecord(total_permute_time, "total_permute_time");
    runtime_csv.addElementToRecord(total_critical_separator_time, "total_critical_sep_time");
    runtime_csv.addElementToRecord(total_critical_permute_time, "total_critical_permute_time");
    runtime_csv.addElementToRecord(ratio_of_sep_to_permute_time, "sep_time/permute_time");
    runtime_csv.addElementToRecord(ratio_of_critical_sep_to_critical_permute_time, "critical_sep_time/critical_permute_time");
    runtime_csv.addElementToRecord(percentage_of_sep_time, "percentage_of_sep_time");
    runtime_csv.addElementToRecord(percentage_of_permute_time, "percentage_of_permute_time");
    runtime_csv.addElementToRecord(percentage_of_critical_sep_time, "percentage_of_critical_sep_time");
    runtime_csv.addElementToRecord(percentage_of_critical_permute_time, "percentage_of_critical_permute_time");
    runtime_csv.addRecord();
    return 0;
}