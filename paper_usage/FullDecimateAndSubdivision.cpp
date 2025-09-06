//
// Created by behrooz zare on 2024-04-28.
//
//
// Created by behrooz zare on 2024-04-06.
//
#include "Parth.h"
#include "Parth_utils.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <igl/false_barycentric_subdivision.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/read_triangle_mesh.h>
#include <igl/avg_edge_length.h>
#include <iostream>
#include <cholmod.h>
#include "remesh_botsch.h"
#include <igl/embree/unproject_onto_mesh.h>
#include <Accelerate/Accelerate.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/upsample.h>
#include <igl/decimate.h>
#include <igl/slice.h>
#include <igl/cotmatrix.h>
#include <igl/unique.h>
#include <igl/setdiff.h>
#include <algorithm>
#include <omp.h>
#include <queue>

Eigen::MatrixXd OV, V;
Eigen::MatrixXi OF, F;
Eigen::VectorXd Z;
Eigen::VectorXi in_prev;
Eigen::VectorXi in_prev_inv;
int DIM = 1;
//int subdivision_step = 5;
double h;
int step = 1;
int num_decimation_steps = 3;
int num_subdivision_steps = 1;
std::vector<int> new_dof_to_old_dof_map;


double total_load_time = 0;
double total_analyze_time = 0;
double total_factor_time = 0;
double total_solve_time = 0;


class AccelerateWrapper { // The class
public:                // Access specifier
    double load_time;
    double analyze_time;
    double factor_time;
    double solve_time;
    double residual;
    double L_NNZ;
    double parth_time;
    int N;

    bool sym_defined;
    bool factor_defined;
    //Accelerate Stuff
    std::vector<long> columnStarts;
    SparseMatrix_Double A;

    SparseOpaqueSymbolicFactorization symbolic_info;
    SparseOpaqueFactorization_Double numeric_info;
    SparseSymbolicFactorOptions opts{};
    SparseStatus_t status;

    PARTH::Parth parth;
    std::vector<int> perm;
    std::vector<int> perm_inv;
    bool activate_parth = false;

    ~AccelerateWrapper() {
        if (sym_defined) {
            SparseCleanup(symbolic_info);
        }

        if (factor_defined) {
            SparseCleanup(numeric_info);
        }

    }

    AccelerateWrapper(bool activate_parth = false) {
        load_time = 0;
        analyze_time = 0;
        factor_time = 0;
        solve_time = 0;
        residual = 0;
        L_NNZ = 0;
        parth_time = 0;
        N = 0;
        sym_defined = false;
        factor_defined = false;

        this->activate_parth = activate_parth;

        if (activate_parth) {
            std::cout << "Activating Immobilizer" << std::endl;
        } else {
            std::cout << "Deactivating Immobilizer" << std::endl;
        }

        parth.setReorderingType(PARTH::ReorderingType::METIS);
        parth.setDOFAddRemoveCauseChange(true);
        parth.setVerbose(true);
        parth.setNDLevels(6);
        parth.setNumberOfCores(10);
    }

    void setMatrix(int *p, int *i, double *x, int A_N, int NNZ, int *Mp, int *Mi,
                   int M_N, std::vector<int> &new_to_old_map) {

        assert(p[A_N] == NNZ);
        this->N = A_N;

        if (new_to_old_map.empty()) {
            parth.setMeshPointers(M_N, Mp, Mi);
        } else {
            parth.setMeshPointers(M_N, Mp, Mi, new_to_old_map);
        }

        const Eigen::Index nColumnsStarts = N + 1;

        columnStarts.resize(nColumnsStarts);

        for (Eigen::Index i = 0; i < nColumnsStarts; i++) columnStarts[i] = p[i];

        SparseAttributes_t attributes{};
        attributes.transpose = false;
        attributes.triangle = SparseLowerTriangle;
        attributes.kind = SparseSymmetric;

        SparseMatrixStructure structure{};
        structure.attributes = attributes;
        structure.rowCount = static_cast<int>(N);
        structure.columnCount = static_cast<int>(N);
        structure.blockSize = 1;
        structure.columnStarts = columnStarts.data();
        structure.rowIndices = const_cast<int *>(i);


        A.structure = structure;
        A.data = const_cast<double *>(x);
    }

    void analyze_pattern(void) {
        analyze_time = omp_get_wtime();
        if (sym_defined) {
            SparseCleanup(symbolic_info);
            sym_defined = false;
        }
        parth_time = omp_get_wtime();
        if (activate_parth) {
            parth.computePermutation(perm, DIM);
        }
        parth_time = omp_get_wtime() - parth_time;

        perm_inv.resize(perm.size());
        for (int i = 0; i < perm.size(); i++) {
            perm_inv[perm[i]] = i;
        }

        opts.control = SparseDefaultControl;
        if (activate_parth) {
            assert(perm.size() == this->N);
            opts.order = perm_inv.data();
            opts.orderMethod = SparseOrderUser;
        } else {
            opts.order = nullptr;
            opts.orderMethod = SparseOrderMetis;
        }

        opts.ignoreRowsAndColumns = nullptr;
        opts.malloc = malloc;
        opts.free = free;
        opts.reportError = nullptr;

        symbolic_info = SparseFactor(SparseFactorizationCholesky, A.structure, opts);
        status = symbolic_info.status;
        if (status != SparseStatusOK) {
            std::cerr << "Symbolic factorization returned with error" << std::endl;
        }
        sym_defined = true;
        analyze_time = omp_get_wtime() - analyze_time;
        L_NNZ = symbolic_info.factorSize_Double / 8;

    }

    bool factorize(void) {
        factor_time = omp_get_wtime();
        if (factor_defined) {
            SparseCleanup(numeric_info);
            factor_defined = false;
        }

        numeric_info = SparseFactor(symbolic_info, A);
        status = numeric_info.status;
        if (status != SparseStatusOK) {
            std::cerr << "Cholesky factorization returned with error" << std::endl;
            return false;
        }
        factor_defined = true;
        factor_time = omp_get_wtime() - factor_time;
        return true;
    }

    void solve(Eigen::VectorXd &rhs, Eigen::VectorXd &result) {
        solve_time = omp_get_wtime();
        if (result.rows() != rhs.rows()) {
            result.resize(rhs.rows());
        }
        DenseVector_Double xmat{};

        xmat.count = result.size();
        xmat.data = result.data();

        DenseVector_Double bmat{};
        bmat.count = rhs.size();
        bmat.data = rhs.data();
        SparseSolve(numeric_info, bmat, xmat);
        solve_time = omp_get_wtime() - solve_time;
    }


    void printTiming() {
        parth.printTiming();
        double total_time = load_time + analyze_time + factor_time + solve_time;
        std::cout << "The analysis time is: " << (analyze_time / total_time) * 100
                  << "%" << std::endl;
        std::cout << "The factorization time is: "
                  << (factor_time / total_time) * 100 << "%" << std::endl;
        std::cout << "The solve time is: " << (solve_time / total_time) * 100 << "%"
                  << std::endl;
        std::cout << "Total Solve time: " << total_time << " seconds" << std::endl;
    }

};


AccelerateWrapper solver(true);

enum SimStage {
    DECIMATION,
    SUBDIVISION,
    FINISHED
};


SimStage stage = SUBDIVISION;

void reset() {
    V = OV;
    F = OF;
    h = igl::avg_edge_length(V, F);
    step = 1;
    stage = DECIMATION;
    new_dof_to_old_dof_map.clear();
    in_prev.resize(0);
    in_prev_inv.resize(0);
}


bool fullSubDivision() {
    int n_prev = V.rows();
    igl::upsample(Eigen::MatrixXd(V), Eigen::MatrixXi(F), V, F);
    new_dof_to_old_dof_map.clear();
    for (int i = 0; i < n_prev; i++) {
        new_dof_to_old_dof_map.push_back(i);
    }

    for (int i = n_prev; i < V.rows(); i++) {
        new_dof_to_old_dof_map.push_back(-1);
    }
    return true;
}


bool fullDecimation() {
    Eigen::MatrixXd V_new;
    Eigen::MatrixXi F_new;
    Eigen::VectorXi F_idx;
    Eigen::VectorXi V_idx;
    int num_faces = 0.7 * F.rows();
    igl::decimate(V, F, num_faces, V_new, F_new, F_idx, V_idx);
    F = F_new;
    V = V_new;
    new_dof_to_old_dof_map.clear();
    for (int i = 0; i < V.rows(); i++) {
        new_dof_to_old_dof_map.push_back(V_idx(i));
    }
    assert(V.rows() == V_idx.rows());
    return true;
}


bool solvingLaplacian() {
    // Find boundary edges
    Eigen::MatrixXi E;
    igl::boundary_facets(F, E);
    // Find boundary vertices
    Eigen::VectorXi b, IA, IC;
    igl::unique(E, b, IA, IC);
    // List of all vertex indices
    Eigen::VectorXi all, in;
    igl::colon<int>(0, V.rows() - 1, all);
    // List of interior indices
    igl::setdiff(all, b, in, IA);

    // Construct and slice up Laplacian
    Eigen::SparseMatrix<double> L, L_in_in, L_in_b;
    igl::cotmatrix(V, F, L);
    igl::slice(L, in, in, L_in_in);
    igl::slice(L, in, b, L_in_b);

    // Dirichlet boundary conditions from z-coordinate
    Z = V.col(2);
    Eigen::VectorXd bc = Z(b);

    // Solve PDE
    Eigen::SparseMatrix<double> mesh_csc =
            PARTH::computeMeshFromHessian(L_in_in, DIM);

    Eigen::SparseMatrix<double> Minus_L_in_in = -L_in_in;


    //Compute the mapping with boundary
    std::vector<int> parth_new_to_old_map(in.rows());
    // Computing the mapping for Parth (due to boundary condition)
    Eigen::VectorXi in_inv(V.rows());
    in_inv.setOnes();
    in_inv = -in_inv;
    if(!new_dof_to_old_dof_map.empty()){
        // computing in_prev_inv
        for (int i = 0; i < in.rows(); i++) {
            in_inv(in(i)) = i;
        }

        for (int parth_dof_idx = 0; parth_dof_idx < in.rows(); parth_dof_idx++) {
            int curr_dof = in(parth_dof_idx);
            int prev_dof = new_dof_to_old_dof_map[curr_dof];

            // If it is a newly added
            if (prev_dof == -1) {
                parth_new_to_old_map[parth_dof_idx] = -1;
                continue;
            }

            // If it is the first time that the function is called
            if (in_prev_inv.rows() == 0) {
                parth_new_to_old_map[parth_dof_idx] = parth_dof_idx;
                continue;
            }

            int curr_dof_pos = in_inv(curr_dof);
            int prev_dof_pos = in_prev_inv(prev_dof);
            assert(parth_dof_idx == curr_dof_pos);
            // If it existed but was not selected before
            if (prev_dof_pos == -1) {
                parth_new_to_old_map[parth_dof_idx] = -1;
                continue;
            }

            // If it existed and selected before
            parth_new_to_old_map[parth_dof_idx] = prev_dof_pos;
        }
    }


    solver.setMatrix(
            Minus_L_in_in.outerIndexPtr(), Minus_L_in_in.innerIndexPtr(),
            Minus_L_in_in.valuePtr(), Minus_L_in_in.rows(), Minus_L_in_in.nonZeros(),
            mesh_csc.outerIndexPtr(), mesh_csc.innerIndexPtr(), mesh_csc.rows(),
            parth_new_to_old_map);

    //Analyze
    solver.analyze_pattern();

    //Factorize
    solver.factorize();

    //Solve
    Eigen::VectorXd rhs = L_in_b * bc;
    Eigen::VectorXd sol;
    solver.solve(rhs, sol);
    Z(in) = sol;

    //Saving CSV


    std::vector<std::string> Runtime_headers;
    Runtime_headers.emplace_back("N");
    Runtime_headers.emplace_back("NNZ");
    Runtime_headers.emplace_back("L_NNZ");

    Runtime_headers.emplace_back("nthreads");

    // Iteration ids
    Runtime_headers.emplace_back("FrameNum");
    Runtime_headers.emplace_back("NewtonIter");

    // Solve Quality
    Runtime_headers.emplace_back("residual");

    // Timing Headers
    Runtime_headers.emplace_back("load_time");
    Runtime_headers.emplace_back("analyze_time");
    Runtime_headers.emplace_back("immobilizer_time");
    Runtime_headers.emplace_back("factor_time");
    Runtime_headers.emplace_back("solve_time");

    // Parth Statistic
    Runtime_headers.emplace_back("contact_size");
    Runtime_headers.emplace_back("parth_reuse");
    Runtime_headers.emplace_back("parth_integration_time");
    Runtime_headers.emplace_back("parth_change_det_time");
    Runtime_headers.emplace_back("parth_perm_comp_time");
    Runtime_headers.emplace_back("parth_map_time");
    Runtime_headers.emplace_back("parth_total_time_time");

    Runtime_headers.emplace_back("total_load_time");
    Runtime_headers.emplace_back("total_analyze_time");
    Runtime_headers.emplace_back("total_factor_time");
    Runtime_headers.emplace_back("total_solve_time");

    std::string Data_name;
    Data_name = "/Users/behroozzare/Desktop/Graphics/ParthSolverDev/output/DOFChangeTest/decimationResults";
    PARTH::CSVManager runtime_csv(Data_name, "some address", Runtime_headers,
                                  false);

    runtime_csv.addElementToRecord(Minus_L_in_in.rows(), "N");
    runtime_csv.addElementToRecord(Minus_L_in_in.nonZeros(), "NNZ");
    runtime_csv.addElementToRecord(solver.L_NNZ, "L_NNZ");
    runtime_csv.addElementToRecord(4, "nthreads");

    // Iteration ids
    runtime_csv.addElementToRecord(0, "FrameNum");
    runtime_csv.addElementToRecord(step, "NewtonIter");

    // Solve Quality
    double residual =
            (rhs - Minus_L_in_in.selfadjointView<Eigen::Lower>() * sol).norm();
    runtime_csv.addElementToRecord(residual, "residual");

    // Timing Headers
    runtime_csv.addElementToRecord(solver.load_time, "load_time");
    runtime_csv.addElementToRecord(solver.analyze_time,
                                   "analyze_time");
    runtime_csv.addElementToRecord(solver.parth_time,
                                   "immobilizer_time");
    runtime_csv.addElementToRecord(solver.factor_time,
                                   "factor_time");
    runtime_csv.addElementToRecord(solver.solve_time,
                                   "solve_time");

    total_load_time += solver.load_time;
    total_analyze_time += solver.analyze_time;
    total_factor_time += solver.factor_time;
    total_solve_time += solver.solve_time;

    // 3 Region STATISICS
    runtime_csv.addElementToRecord(solver.parth.getNumChanges(), "contact_size");
    runtime_csv.addElementToRecord(solver.parth.getReuse(), "parth_reuse");
    runtime_csv.addElementToRecord(solver.parth.dof_change_integrator_time,
                                   "parth_integration_time");
    runtime_csv.addElementToRecord(solver.parth.change_computation_time,
                                   "parth_change_det_time");
    runtime_csv.addElementToRecord(solver.parth.compute_permutation_time,
                                   "parth_perm_comp_time");
    runtime_csv.addElementToRecord(solver.parth.map_mesh_to_matrix_computation_time,
                                   "parth_map_time");
    double parth_total_time = solver.parth.dof_change_integrator_time +
                              solver.parth.change_computation_time +
                              solver.parth.compute_permutation_time +
                              solver.parth.map_mesh_to_matrix_computation_time;
    runtime_csv.addElementToRecord(parth_total_time,
                                   "parth_total_time_time");


    runtime_csv.addElementToRecord(total_load_time, "total_load_time");
    runtime_csv.addElementToRecord(total_analyze_time, "total_analyze_time");
    runtime_csv.addElementToRecord(total_factor_time, "total_factor_time");
    runtime_csv.addElementToRecord(total_solve_time, "total_solve_time");

    runtime_csv.addRecord();


    //Printing Timing
    solver.printTiming();
    //Reset the timings
    solver.analyze_time = 0;
    solver.factor_time = 0;
    solver.solve_time = 0;
    solver.load_time = 0;
    in_prev = in;
    in_prev_inv = in_inv;
    return true;
}

void OneStepSim() {
    if (stage == DECIMATION && step % num_decimation_steps == 0) {
        stage = FINISHED;
    } else if (stage == SUBDIVISION && step % num_subdivision_steps == 0) {
        stage = FINISHED;
    }

    if (stage == DECIMATION) {
        fullDecimation();
        step++;
    } else if (stage == SUBDIVISION) {
        fullSubDivision();
        step++;
    } else {
        std::cerr << "UNKNOWN STAGE" << std::endl;
    }
    solvingLaplacian();

}

bool remesh() {
    Eigen::VectorXd target;
    int n = V.rows();
    target = Eigen::VectorXd::Constant(n, h * step);
    remesh_botsch(V, F);
    step++;
    return true;
}

void UpdateViewer(igl::opengl::glfw::Viewer &viewer) {
    viewer.data().clear();
    viewer.data().set_mesh(V, F);
    viewer.data().set_face_based(true);
    if (Z.rows() == V.rows()) {
        viewer.data().set_data(Z);
    }
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
    F = OF;
    h = igl::avg_edge_length(V, F);
    igl::opengl::glfw::Viewer viewer;


    viewer.callback_key_pressed =
            [&](igl::opengl::glfw::Viewer &viewer, unsigned int key, int mods) -> bool {
                switch (key) {
                    default:
                        return false;
                    case ' ':
                        viewer.core().is_animating = !viewer.core().is_animating;
                        if (viewer.core().is_animating) {
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
                        OneStepSim();
                        UpdateViewer(viewer);
                        return true;
                    case 'p':
                        solver.activate_parth = !solver.activate_parth;
                        if (solver.activate_parth) {
                            std::cout << "Activating Parth" << std::endl;
                        } else {
                            std::cout << "Deactivating Parth" << std::endl;
                        }
                }
                return false;
            };


    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer &viewer) -> bool {
        if (viewer.core().is_animating) {
            OneStepSim();
            UpdateViewer(viewer);
            if (stage == FINISHED) {
                viewer.core().is_animating = false;
            }
        }
        return false;
    };


    viewer.core().is_animating = true;
    UpdateViewer(viewer);
    viewer.core().is_animating = false;
    viewer.launch();


    return 0;
}