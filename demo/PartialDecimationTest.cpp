//
// Created by behrooz zare on 2024-04-28.
//
//
// Created by behrooz zare on 2024-04-06.
//
#include "Parth.h"
#include "GIF.h"
#include "CLI/CLI.hpp"
#include "Parth_utils.h"
#include "randomPatch.h"
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
#include "projected_cotmatrix.h"
#include <igl/unique.h>
#include <igl/setdiff.h>
#include <algorithm>
#include <omp.h>
#include <queue>

Eigen::MatrixXd OV, V;
Eigen::MatrixXi OF, F;
Eigen::VectorXd Z;
bool init = false;
Eigen::VectorXi in_init_prev_inv;
Eigen::VectorXi in_init_prev;
Eigen::VectorXi patch;
int step = 0;
int DIM = 1;


double total_load_time = 0;
double total_analyze_time = 0;
double total_factor_time = 0;
double total_solve_time = 0;


// Gif variables
GifWriter GIFWriter;
uint32_t GIFDelay = 100; //*10ms
int GIFStep = 1;
double GIFScale = 0.6;
std::string gif_address;


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

void reset() {
    V = OV;
    F = OF;
    solver.parth.clearParth();
    patch.resize(0);
}


void DecimateThePatch(Eigen::VectorXi &I, Eigen::MatrixXi &F,
                      Eigen::MatrixXd &V, Eigen::MatrixXi &F_dec, Eigen::MatrixXd &V_dec,
                      std::vector<int> &new_to_old_dof_map) {
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

#ifndef NDEBUG
    // Find boundary edges
    igl::boundary_facets(F_sub, E);
    // Find boundary vertices
    Eigen::VectorXi b_new;
    igl::unique(E, b_new, IA, IC);
    assert(b_new.rows() == b.rows());
    for (int i = 0; i < b.rows(); i++) {
        assert(b(i) == b_new(i));
    }
#endif

    double h = igl::avg_edge_length(V, F);
    assert(b.rows() != V_sub.rows());
    std::cout << "Number of faces " << F_sub.rows() << std::endl;
    std::cout << "Number of vertices " << V_sub.rows() << std::endl;
    remesh_botsch(V_sub, F_sub, h * 5, 10, b, false);
    std::cout <<"Remesher effect " << std::endl;
    std::cout << "Number of faces " << F_sub.rows() << std::endl;
    std::cout << "Number of vertices " << V_sub.rows() << std::endl;
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
    patch.resize(patch_indices.size());
    for (int i = 0; i < patch.rows(); i++) {
        patch(i) = patch_indices[i];
    }
}

bool solvingLaplacian(std::vector<int> &new_to_old_dof_mapping, std::string csv_address, bool write_csv) {
    double end_to_end_time = omp_get_wtime();
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
    PARTHDEMO::projected_cotmatrix(V, F, L);
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
    Eigen::VectorXi in_inv(V.rows());
    in_inv.setOnes();
    in_inv = -in_inv;
    // computing in_prev_inv
    for (int i = 0; i < in.rows(); i++) {
        in_inv(in(i)) = i;
    }

    if (!new_to_old_dof_mapping.empty()) {
        for (int parth_dof_idx = 0; parth_dof_idx < in.rows(); parth_dof_idx++) {
            int curr_dof = in(parth_dof_idx);
            int prev_dof = new_to_old_dof_mapping[curr_dof];

            // If it is a newly added
            if (prev_dof == -1) {
                parth_new_to_old_map[parth_dof_idx] = -1;
                continue;
            }

            // If it is the first time that the function is called
            if (in_init_prev_inv.rows() == 0) {
                parth_new_to_old_map[parth_dof_idx] = parth_dof_idx;
                continue;
            }

            int curr_dof_pos = in_inv(curr_dof);
            int prev_dof_pos = in_init_prev_inv(prev_dof);
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


    end_to_end_time = omp_get_wtime() - end_to_end_time;

    //Saving CSV
    if (write_csv) {
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

        Runtime_headers.emplace_back("end_to_end_time");


        std::string Data_name;
        PARTH::CSVManager runtime_csv(csv_address, "some address", Runtime_headers,
                                      false);

        runtime_csv.addElementToRecord(Minus_L_in_in.rows(), "N");
        runtime_csv.addElementToRecord(Minus_L_in_in.nonZeros(), "NNZ");
        runtime_csv.addElementToRecord(solver.L_NNZ, "L_NNZ");
        runtime_csv.addElementToRecord(4, "nthreads");

        // Iteration ids
        runtime_csv.addElementToRecord(0, "FrameNum");
        runtime_csv.addElementToRecord(step++, "NewtonIter");

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
        runtime_csv.addElementToRecord(end_to_end_time, "end_to_end_time");

        runtime_csv.addRecord();

    }


    //Printing Timing
    solver.printTiming();
    //Reset the timings
    solver.analyze_time = 0;
    solver.factor_time = 0;
    solver.solve_time = 0;
    solver.load_time = 0;
    if (!init) {
        in_init_prev = in;
        in_init_prev_inv = in_inv;
        init = true;
    }

    return true;
}

void UpdateViewer(igl::opengl::glfw::Viewer &viewer) {
    viewer.data().clear();
    viewer.data().set_mesh(V, F);
    viewer.data().set_face_based(true);
    Eigen::MatrixXd C = Eigen::MatrixXd::Constant(F.rows(), 3, 1);
//    Compute the adjacency of a face
    for (int i = 0; i < patch.rows(); i++) {
        C.row(patch(i)) << 1, 0, 0;
    }
    viewer.data().set_colors(C);

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

// GIF Wrapper
void initGif(igl::opengl::glfw::Viewer &viewer, std::string save_address) {
    GifBegin(&GIFWriter, save_address.c_str(),
             (viewer.core().viewport[2] - viewer.core().viewport[0]),
             (viewer.core().viewport[3] - viewer.core().viewport[1]), GIFDelay);
}

void writeGif(igl::opengl::glfw::Viewer &viewer) {
    int width = static_cast<int>((viewer.core().viewport[2] - viewer.core().viewport[0]));
    int height = static_cast<int>((viewer.core().viewport[3] - viewer.core().viewport[1]));

    // Allocate temporary buffers for image
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> R(width, height);
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> G(width, height);
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> B(width, height);
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> A(width, height);

    // Draw the scene in the buffers
    viewer.core().draw_buffer(viewer.data(), false, R, G, B, A);

    std::vector<uint8_t> img(width * height * 4);
    for (int rowI = 0; rowI < width; rowI++) {
        for (int colI = 0; colI < height; colI++) {
            int indStart = (rowI + (height - 1 - colI) * width) * 4;
            img[indStart] = R(rowI, colI);
            img[indStart + 1] = G(rowI, colI);
            img[indStart + 2] = B(rowI, colI);
            img[indStart + 3] = A(rowI, colI);
        }
    }

    GifWriteFrame(&GIFWriter, img.data(), width, height, GIFDelay);
}


namespace fs = std::filesystem;

struct CLIArgs {
    std::string input;
    std::string output;
    std::string simName = "test";
    std::string solveLib = "CHOLMOD";
    int immobilizer = 0;
    int numThreads = 10;

    CLIArgs(int argc, char *argv[]) {
        CLI::App app{"Parth Solver"};

        app.add_option("-o,--output", output, "output folder name");
        app.add_option("-i,--input", input, "input mesh name");
        app.add_option("--SimName", simName, "Simulation name");
        app.add_option("--IM", immobilizer, "ADD PARTH IMMOBILIZER");
        app.add_option("--numThreads", numThreads,
                       "maximum number of threads to use")
                ->default_val(numThreads);
        app.add_option("--SolverType", solveLib,
                       "Choose one of the solvers CHOLMOD, EIGEN, MKL, STRUMPACK");
        try {
            app.parse(argc, argv);
        } catch (const CLI::ParseError &e) {
            exit(app.exit(e));
        }
    }
};


int main(int argc, char *argv[]) {

    CLIArgs args(argc, argv);

    igl::read_triangle_mesh(args.input, OV, OF);
    V = OV;
    F = OF;

    //Compute 100 random center of patches
    std::vector<int> fid_list;
    for (int i = 0; i < 50; i++) {
        fid_list.emplace_back(rand() % F.rows());
    }

    std::vector<int> ring_list{4};
//
    std::string Data_name = args.output + "accelerate_parth_" + args.simName;
    std::vector<int> tmp;
    solvingLaplacian(tmp, Data_name, true);
    solver.activate_parth = true;

    //Main loop for experiment -> Parth Timing
    for(int r_size = 0; r_size < ring_list.size(); r_size++){
        for (int test_cnt = 0; test_cnt < fid_list.size(); test_cnt++) {
            std::cout << "----------------- Test number: " << test_cnt << " -----------------" << std::endl;
            reset();
            solvingLaplacian(tmp, Data_name, false); //Init Parth datastructure
            Eigen::MatrixXi F_up;
            Eigen::MatrixXd V_up;

            //------------------- Randomly select a Patch -------------------
            PARTHDEMO::randomPatch(fid_list[test_cnt], ring_list[r_size], patch, F, V);
            //------------------- Increase the dofs in that patch -------------------
            std::vector<int> new_to_old_map;
            DecimateThePatch(patch, F, V, F_up, V_up, new_to_old_map);
            //Compute the mapping

            V = V_up;
            F = F_up;
            //Solving the laplacian
            solvingLaplacian(new_to_old_map, Data_name, true);
        }
    }

//
//
    total_load_time = 0;
    total_analyze_time = 0;
    total_factor_time = 0;
    total_solve_time = 0;
    //Base Computation with activating Parth
    Data_name = args.output + "accelerate_" + args.simName;
    tmp.clear();
    reset();
    solvingLaplacian(tmp, Data_name, true);
    solver.activate_parth = false;

    //Main loop for experiment -> Parth Timing
    for(int r_size = 0; r_size < ring_list.size(); r_size++){
        for (int test_cnt = 0; test_cnt < fid_list.size(); test_cnt++) {
            std::cout << "----------------- Test number: " << test_cnt << " -----------------" << std::endl;
            reset();
            solvingLaplacian(tmp, Data_name, false); //Init Parth datastructure
            Eigen::MatrixXi F_up;
            Eigen::MatrixXd V_up;

            //------------------- Randomly select a Patch -------------------
            PARTHDEMO::randomPatch(fid_list[test_cnt], ring_list[r_size], patch, F, V);
            //------------------- Increase the dofs in that patch -------------------
            std::vector<int> new_to_old_map;
            DecimateThePatch(patch, F, V, F_up, V_up, new_to_old_map);

            V = V_up;
            F = F_up;
            //Solving the laplacian
            solvingLaplacian(new_to_old_map, Data_name, true);
        }
    }
//
//
    igl::opengl::glfw::Viewer viewer;

    int frame = 0;
    int ring_size = 0;
    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer &viewer) -> bool {
        if (frame == fid_list.size()) {
            viewer.core().is_animating = false;
        } else {
            viewer.core().is_animating = true;
        }
        if (viewer.core().is_animating) {
            Eigen::MatrixXi F_dec;
            Eigen::MatrixXd V_dec;
            //------------------- Randomly select a Patch -------------------
            PARTHDEMO::randomPatch(fid_list[frame], ring_list[ring_size++], patch, F, V);
            //------------------- Increase the dofs in that patch -------------------
            std::vector<int> new_to_old_mapping;
            DecimateThePatch(patch, F, V, F_dec, V_dec, new_to_old_mapping);
            F = F_dec;
            V = V_dec;
            UpdateViewer(viewer);

            if (ring_size % ring_list.size() == 0) {
                frame++;
                ring_size = 0;
            }
        }

        return false;
    };

    bool init_gif = false;
    viewer.callback_post_draw = [&](igl::opengl::glfw::Viewer &viewer) -> bool {
        if (viewer.core().is_animating) {
            if (!init_gif) {
                initGif(viewer, args.output + "out.gif");
                init_gif = true;
            }
            writeGif(viewer);
        } else {
            GifEnd(&GIFWriter);
//            viewer.data().clear();
            return true;
        }
        F = OF;
        V = OV;
        return false;
    };


    UpdateViewer(viewer);
    viewer.core().is_animating = true;
    viewer.launch();
    return 0;
}