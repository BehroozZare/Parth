//
// Created by behrooz zare on 2024-01-04.
//
#include <iostream>
#include <unsupported/Eigen/SparseExtra>
#include <CLI/CLI.hpp>
#include <Barb.h>
#include <string>
#include <NewtonIterVisualization.h>
#include <cmath>
#include <queue>


namespace fs = std::filesystem;

struct CLIArgs {
    std::string input;
    std::string simName = "test";
    int frame;
    int iter;
    int immobilizer = 0;
    int numThreads = 4;

    CLIArgs(int argc, char *argv[]) {
        CLI::App app{"Parth Solver"};

        app.add_option("-i,--input", input, "input folder name");
        app.add_option("--SimName", simName, "Simulation name");
        app.add_option("--frame", frame, "Frame number");
        app.add_option("--iter", iter, "Iter number");

        try {
            app.parse(argc, argv);
        } catch (const CLI::ParseError &e) {
            exit(app.exit(e));
        }
    }
};


class ParthWrapper { // The class
public:
    PARTH::ParthSolver ParthImmobilizer;
    std::vector<int> perm;
    std::vector<int> node_to_group;
    std::vector<int> group_to_node;

    int N;

    ~ParthWrapper() {
    }

    ParthWrapper(int num_threads = 4, int num_levels = 7) {
        ParthImmobilizer.Options().setComputeResidual(true);
        ParthImmobilizer.Options().setVerbose(true);
        ParthImmobilizer.Options().setAnalyzeType(
                PARTH::ParthSolver::AnalyzeType::SimulationAware);
        ParthImmobilizer.Options().setReorderingType(
                PARTH::ParthSolver::ReorderingType::METIS);
        ParthImmobilizer.Options().setNumericReuseType(
                PARTH::ParthSolver::NumericReuseType::NUMERIC_REUSE_PARALLEL);
        std::cout << "Running with " << omp_get_max_threads() << std::endl;
        ParthImmobilizer.Options().setNumberOfCores(num_threads);
        ParthImmobilizer.Options().setNumRegions(num_levels);
    }

    void setMatrix(int *p, int *i, double *x, int A_N, int NNZ, int *Mp, int *Mi,
                   int M_N) {
        this->N = A_N;
        ParthImmobilizer.setMatrixPointers(p, i, x, A_N, NNZ);
        ParthImmobilizer.setMeshPointers(M_N, Mp, Mi);
    }

    void analyze(void) {
        ParthImmobilizer.getPermutationWithReuse(perm);
        assert(perm.size() == this->N);
    }


};

int getLvl(int idx){
    int tmp = idx;
    int lvl = 0;
    while(tmp != 0){
        if(tmp % 2 == 1){
            tmp = (tmp - 1) / 2;
        }else{
            tmp = (tmp - 2) / 2;
        }
        lvl++;
    }
    return lvl;
}

std::vector<int> getAllChild(int idx, int max_lvl){
    std::vector<int> all_child;
    std::queue<int> child_set;
    child_set.push(idx * 2 + 1);
    child_set.push(idx * 2 + 2);
    while(!child_set.empty()){
        int current = child_set.front();
        if(getLvl(current) <= max_lvl){
            all_child.push_back(current);
            child_set.push(current * 2 + 1);
            child_set.push(current * 2 + 2);
        }
        child_set.pop();
    }
    return all_child;
}

std::vector<int> getChildToParentMap(int current_lvl, int max_lvl){
    int total_nodes = std::pow(2, max_lvl + 1) - 1;
    std::vector<int> result(total_nodes);
    for(int i = 0; i < result.size(); i++){
        result[i] = i;
    }
    if(current_lvl == max_lvl){
        return result;
    }
    //The code is just like that :))
    int start_idx = std::pow(2, current_lvl) - 1;
    int end_idx = std::pow(2, current_lvl) + start_idx;


    for(int idx = start_idx; idx < end_idx; idx++){
        std::vector<int> children = getAllChild(idx, max_lvl);
        for(auto& ch : children){
            result[ch] = idx;
        }
    }
    return result;
}


int main(int argc, char *argv[]) {
    CLIArgs args(argc, argv);

    omp_set_num_threads(args.numThreads);

    bool immo = false;
    auto list_of_files = PARTH::getFileNames(args.input);
    std::vector<std::tuple<int, int>> list_of_iter;
    for (auto &x: list_of_files) {
        auto s = PARTH::split_string(x, "_");
        if (s.empty() || s[0] != "V" || s[1] == "rest") {
            continue;
        }
        list_of_iter.emplace_back(
                std::tuple<int, int>(std::stoi(s[1]), std::stoi(s[2])));
    }

    // Sort the list of iterations
    std::sort(list_of_iter.begin(), list_of_iter.end(), PARTH::sortbyFirstAndSec);
    list_of_iter.erase(std::unique(list_of_iter.begin(), list_of_iter.end()),
                       list_of_iter.end());

    int cnt = 0;

    PARTH::NewtonIterVisualization vis_obj = PARTH::NewtonIterVisualization();
    vis_obj.init("arma", true,
                 "../../output/");

    int num_levels =7;

    ParthWrapper parth_obj(args.numThreads, num_levels);

    // Load the SF matrix
    Eigen::MatrixXi SF = PARTH::openMatInt(args.input + "SF");

    // Load the F matrix
    Eigen::MatrixXi F = PARTH::openMatInt(args.input + "F");

    // Load the V_rest matrix
    Eigen::MatrixXd V_rest = PARTH::openMatDouble(args.input + "V_rest");

    // Main loop for analyzing iterations of a frame
    for (auto &iter: list_of_iter) {
        if (iter == list_of_iter.back()) {
            break;
        }

        cnt++;


        int frame = std::get<0>(iter);
        int iteration = std::get<1>(iter);

//        if(frame != 0){
//            if(frame != 10){
//                continue;
//            }
//        }
//        if(iteration != 0){
//            continue;
//        }
        std::cout << "Testing for Frame " << frame << " and Iteration " << iteration
                  << std::endl;

        //************** Load the Mesh and hessian *************
        Eigen::SparseMatrix<double> mesh_csc;
        std::string mesh_name = args.input + "mesh_" +
                                std::to_string(std::get<0>(iter)) + "_" +
                                std::to_string(std::get<1>(iter)) + "_" + "IPC.mtx";
        if (!Eigen::loadMarket(mesh_csc, mesh_name)) {
            std::cerr << "File " << mesh_name << " is not found" << std::endl;
        }

        Eigen::SparseMatrix<double> lower_A_csc;
        std::string hessian_name =
                args.input + "hessian_" + std::to_string(std::get<0>(iter)) + "_" +
                std::to_string(std::get<1>(iter)) + "_" + "last_IPC.mtx";

        if (!Eigen::loadMarket(lower_A_csc, hessian_name)) {
            std::cerr << "File " << hessian_name << " is not found" << std::endl;
        }

        Eigen::MatrixXd V_curr = PARTH::openMatDouble(
                args.input + "V_" + std::to_string(std::get<0>(iter)) + "_" +
                std::to_string(std::get<1>(iter)));

        //***********************************************************************
        Eigen::VectorXd Parth_sol(lower_A_csc.rows());
        Parth_sol.setZero();

        parth_obj.setMatrix(
                lower_A_csc.outerIndexPtr(), lower_A_csc.innerIndexPtr(),
                lower_A_csc.valuePtr(), lower_A_csc.rows(), lower_A_csc.nonZeros(),
                mesh_csc.outerIndexPtr(), mesh_csc.innerIndexPtr(), mesh_csc.rows());

        std::cout << "++++++++++++++++++ Parth: Analysing *********************"
                  << std::endl;
        parth_obj.analyze();

        // Load positions
        Eigen::MatrixXd V = PARTH::openMatDouble(
                args.input + "V_" + std::to_string(std::get<0>(iter)) + "_" +
                std::to_string(std::get<1>(iter)));


        std::vector<int> group = parth_obj.ParthImmobilizer.getElemRegions();
        if(group.size() != V.rows()){
            std::cerr << "The group vector must have equal size to the number of rows in V" << std::endl;
        }
        std::cout << "Reuse: " << parth_obj.ParthImmobilizer.regions.getReuseRatio() <<
        " Contacts: "<< parth_obj.ParthImmobilizer.regions.num_contact_points << std::endl;
        int total_num_groups = parth_obj.ParthImmobilizer.regions.tree_nodes.size();
        std::vector<int> dirty_mesh_nodes = parth_obj.ParthImmobilizer.regions.getDirtyNodes();

        for(auto& iter: dirty_mesh_nodes){
            group[iter] = -1;
        }
//        std::vector<double> group_ratio(total_num_groups, 0);
//        std::vector<int> group_tmp = group;


//        for(auto& g : group){
//            group_ratio[g]++;
//        }
//        for(int g = 0; g < group_ratio.size(); g++){
//            std::cout << "Group " << g << ": " << group_ratio[g] / group.size() << std::endl;
//        }

//        for(int l = 0; l <= num_levels; l++){
//            int num_groups = std::pow(2,l + 1) - 1;
//            std::vector<std::string> group_name(num_groups);
//            for(int i = 0; i < num_groups; i++){
//                group_name[i] = "L" + std::to_string(l) + "_g" + std::to_string(i) +
//                                "_" + std::to_string(frame) + "_" + std::to_string(iteration);
//            }
//            std::vector<int> mapping = getChildToParentMap(l, num_levels);
//
//            for(int i = 0; i < group_tmp.size(); i++){
//                group_tmp[i] = mapping[group[i]];
//            }
//            std::fill(group_ratio.begin(), group_ratio.end(), 0);
//            std::cout << "=============== " << l << " ============" << std::endl;
//            for(auto& g : group_tmp){
//                group_ratio[g]++;
//            }
//            for(int g = 0; g < group_ratio.size(); g++){
//                std::cout << "Group " << g << ": " << group_ratio[g] / group_tmp.size() << std::endl;
//            }
//            PARTH::WriteObjWithGroup(V, F, group_tmp, group_name, "../../output/");
//
//        }
        PARTH::applyTextureBasedOnLeavesAndSeparator(V, SF, num_levels,
                                                     group,
                                                     "../../output/", frame, iteration);
        //Save Permutation matrix
        std::ofstream file;
        file.open("../../output/perm_" + std::to_string(frame) + "_" + std::to_string(iteration) + ".txt");
        if(file.is_open()){
            file << parth_obj.perm.size() << "\n"; // write the vector size on the first line.
            for(auto &n : parth_obj.perm){ // iterate over the vector and write each number on a new line.
                file << n << "\n";
            }
            file.close();
        }else{
            std::cout << "Unable to open file";
        }


        std::fill(group.begin(), group.end(), 0);

        vis_obj.addIterSol(V, parth_obj.ParthImmobilizer.getElemRegions().data(),
                           std::get<0>(iter), std::get<1>(iter));

    }


    vis_obj.visualizeNewtonIters(V_rest, SF,
                                 parth_obj.ParthImmobilizer.regions.tree_nodes.size());

    return 0;
}