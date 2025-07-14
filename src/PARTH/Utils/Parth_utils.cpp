//
// Created by behrooz on 2022-04-26.
//

#include "includes/Parth_utils.h"
#include <igl/boundary_facets.h>

namespace PARTH {

    Eigen::VectorXd flattenVector(const Eigen::MatrixXd &x) {
        Eigen::VectorXd flatten_x(x.rows() * 3);
        for (int vI = 0; vI < x.rows(); vI++) {
            flatten_x.segment<3>(vI * 3) = x.row(vI).transpose();
        }
        return flatten_x;
    }

// ========== OTHER FUNCTIONS ===========
    void printEigenMatrix(Eigen::MatrixXd &V, long max_row_cnt) {
        std::string sep = "\n----------------------------------------\n";
        auto sub_V = V.block(0, 0, std::min(max_row_cnt, V.rows()), V.cols());
        Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
        std::cout << sub_V.format(CleanFmt) << sep;
    }

    void printEigenMatrix(Eigen::VectorXd &V, long max_row_cnt) {
        std::string sep = "\n----------------------------------------\n";
        auto sub_V = V.block(0, 0, std::min(max_row_cnt, V.rows()), V.cols());
        Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
        std::cout << sub_V.format(CleanFmt) << sep;
    }

    void printEigenMatrix(Eigen::VectorXi &V, long max_row_cnt) {
        std::string sep = "\n----------------------------------------\n";
        auto sub_V = V.block(0, 0, std::min(max_row_cnt, V.rows()), V.cols());
        Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
        std::cout << sub_V.format(CleanFmt) << sep;
    }

    void compute3DPossion(Eigen::SparseMatrix<double, Eigen::RowMajor> &A, int n) {
        int n2 = n * n;
        int N = n * n2;
        int nnz = 7 * N - 6 * n2;
        A.conservativeResize(N, N);
        A.reserve(nnz);
        int *cptr = A.outerIndexPtr();
        int *rind = A.innerIndexPtr();
        double *val = A.valuePtr();
        nnz = 0;
        cptr[0] = 0;
        for (int xdim = 0; xdim < n; xdim++) {
            for (int ydim = 0; ydim < n; ydim++) {
                for (int zdim = 0; zdim < n; zdim++) {
                    int ind = zdim + ydim * n + xdim * n2;
                    val[nnz] = 6.0;
                    rind[nnz++] = ind;
                    if (zdim > 0) {
                        val[nnz] = -1.0;
                        rind[nnz++] = ind - 1;
                    } // left
                    if (zdim < n - 1) {
                        val[nnz] = -1.0;
                        rind[nnz++] = ind + 1;
                    } // right
                    if (ydim > 0) {
                        val[nnz] = -1.0;
                        rind[nnz++] = ind - n;
                    } // front
                    if (ydim < n - 1) {
                        val[nnz] = -1.0;
                        rind[nnz++] = ind + n;
                    } // back
                    if (xdim > 0) {
                        val[nnz] = -1.0;
                        rind[nnz++] = ind - n2;
                    } // up
                    if (xdim < n - 1) {
                        val[nnz] = -1.0;
                        rind[nnz++] = ind + n2;
                    } // down
                    cptr[ind + 1] = nnz;
                }
            }
        }
    }

///---------------------------------------------------------------------------------------\n
/// create a list of files in a folder.
///---------------------------------------------------------------------------------------\n
/// @return A list of names of files in a folder
    std::vector<std::string>
    getFileNames(std::string path ///<[in] the address of the folder
    ) {
        std::vector<std::string> list_of_files;
        for (const auto &entry: std::filesystem::directory_iterator(path)) {
            list_of_files.emplace_back(entry.path().filename());
        }
        return list_of_files;
    }

///---------------------------------------------------------------------------------------\n
/// split a string based on the delimiter\n
///---------------------------------------------------------------------------------------\n
/// @return a vector of string\n
    std::vector<std::string>
    split_string(std::string input,    ///<[in] the address of the folder
                 std::string delimiter ///<[in] the address of the folder
    ) {
        std::vector<std::string> splitted_string;
        size_t pos = 0;
        std::string token;
        while ((pos = input.find(delimiter)) != std::string::npos) {
            token = input.substr(0, pos);
            splitted_string.emplace_back(token);
            input.erase(0, pos + delimiter.length());
        }
        splitted_string.emplace_back(input);
        return splitted_string;
    }

    bool sortbyFirstAndSec(const std::tuple<int, int> &a,
                           const std::tuple<int, int> &b) {
        if (std::get<0>(a) != std::get<0>(b)) {
            return std::get<0>(a) < std::get<0>(b);
        } else {
            return (std::get<1>(a) < std::get<1>(b));
        }
    }

///---------------------------------------------------------------------------------------\n
/// Compute dot product of two vector without overflow
///---------------------------------------------------------------------------------------\n
    double dotProduct(double *a, double *b, int size) {
        double dot_product = 0;
        for (int iter = 0; iter < size; iter++) {
            dot_product += (a[iter] * b[iter]);
        }
        return dot_product;
    }

///---------------------------------------------------------------------------------------\n
/// Compute the inf norm of matrix A in CSC/CSR format (max(A[i]) for 0 <= i < n
///---------------------------------------------------------------------------------------\n
    double
    computeInfNormOfMatrixA(double *Ax, ///<[in] Value vector in CSC/CSR format
                            int size    ///[in] nnz in A
    ) {
        double max = 0;
        for (int i = 0; i < size; i++) {
            if (abs(Ax[i] > max)) {
                max = Ax[i];
            }
        }
        return max;
    }

    void vectorSum(double *a, double a_scale, double *b, double b_scale, int size,
                   double *result) {
#pragma omp parallel for default(none)                                         \
    shared(size, result, a_scale, a, b_scale, b)
        for (int iter = 0; iter < size; iter++) {
            result[iter] = a_scale * a[iter] + b_scale * b[iter];
        }
    }


    Eigen::SparseMatrix<double> computeMeshFromHessian(Eigen::SparseMatrix<double> &A, int DIM) {
        Eigen::SparseMatrix<double> mesh_csc;
        assert(A.rows() % DIM == 0);
        mesh_csc.resize(A.rows() / DIM, A.cols() / DIM);
        std::vector<Eigen::Triplet<double>> coefficients;
        int N = A.rows();
        int *Ap = A.outerIndexPtr();
        int *Ai = A.innerIndexPtr();

        for (int c = 0; c < N; c += 3) {
            assert((Ap[c + 1] - Ap[c]) % 3 == 0);
            for (int r_ptr = Ap[c]; r_ptr < Ap[c + 1]; r_ptr += 3) {
                int r = Ai[r_ptr];
                int mesh_c = c / DIM;
                int mesh_r = r / DIM;
                if (mesh_c != mesh_r) {
                    coefficients.emplace_back(mesh_c, mesh_r, 1);
                    coefficients.emplace_back(mesh_r, mesh_c, 1);
                }
            }
        }

        mesh_csc.setFromTriplets(coefficients.begin(), coefficients.end());
        assert((A.nonZeros() * 2 - A.rows()) / (DIM * DIM) - mesh_csc.rows() == mesh_csc.nonZeros());
        return mesh_csc;
    }


///---------------------------------------------------------------------------------------\n
/// Compute norm 2 of two vector without overflow
///---------------------------------------------------------------------------------------\n
    double Norm2(Eigen::VectorXi &a) {
        assert(a.size() != 0);
        double norm2 = 0;
        for (int iter = 0; iter < a.size(); iter++) {
            norm2 += (a[iter] * a[iter]);
        }
        return std::sqrt(norm2);
    }

    // the inspiration for creating this function was drawn from here (I did NOT
    // copy and paste the code)
    // https://stackoverflow.com/questions/34247057/how-to-read-csv-file-and-assign-to-eigen-matrix
    Eigen::MatrixXd openMatDouble(std::string fileToOpen) {
        // the inspiration for creating this function was drawn from here (I did NOT
        // copy and paste the code)
        // https://stackoverflow.com/questions/34247057/how-to-read-csv-file-and-assign-to-eigen-matrix

        // the input is the file: "fileToOpen.csv":
        // a,b,c
        // d,e,f
        // This function converts input file data into the Eigen matrix format

        // the matrix entries are stored in this variable row-wise. For example if we
        // have the matrix: M=[a b c
        //    d e f]
        // the entries are stored as matrixEntries=[a,b,c,d,e,f], that is the variable
        // "matrixEntries" is a row vector later on, this vector is mapped into the
        // Eigen matrix format
        std::vector<double> matrixEntries;

        // in this object we store the data from the matrix
        std::ifstream matrixDataFile(fileToOpen);
        if (!matrixDataFile.is_open()) {
            std::cerr << "Can not open the matrix in " << fileToOpen << std::endl;
        }

        // this variable is used to store the row of the matrix that contains commas
        std::string matrixRowString;

        // this variable is used to store the matrix entry;
        std::string matrixEntry;

        // this variable is used to track the number of rows
        int matrixRowNumber = 0;

        while (std::getline(
                matrixDataFile,
                matrixRowString)) // here we read a row by row of matrixDataFile and store
            // every line into the string variable matrixRowString
        {
            std::stringstream matrixRowStringStream(
                    matrixRowString); // convert matrixRowString that is a string to a
            // stream variable.

            while (getline(matrixRowStringStream, matrixEntry,
                           ',')) // here we read pieces of the stream
                // matrixRowStringStream until every comma, and store
                // the resulting character into the matrixEntry
            {
                matrixEntries.push_back(std::stod(
                        matrixEntry)); // here we convert the string to double and fill in the
                // row vector storing all the matrix entries
            }
            matrixRowNumber++; // update the column numbers
        }

        // here we convet the vector variable into the matrix and return the resulting
        // object, note that matrixEntries.data() is the pointer to the first memory
        // location at which the entries of the vector matrixEntries are stored;
        return Eigen::Map<
                Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
                matrixEntries.data(), matrixRowNumber,
                matrixEntries.size() / matrixRowNumber);
    }

    // the inspiration for creating this function was drawn from here (I did NOT
    // copy and paste the code)
    // https://stackoverflow.com/questions/34247057/how-to-read-csv-file-and-assign-to-eigen-matrix
    Eigen::MatrixXi openMatInt(std::string fileToOpen) {

        // the inspiration for creating this function was drawn from here (I did NOT
        // copy and paste the code)
        // https://stackoverflow.com/questions/34247057/how-to-read-csv-file-and-assign-to-eigen-matrix

        // the input is the file: "fileToOpen.csv":
        // a,b,c
        // d,e,f
        // This function converts input file data into the Eigen matrix format

        // the matrix entries are stored in this variable row-wise. For example if we
        // have the matrix: M=[a b c
        //    d e f]
        // the entries are stored as matrixEntries=[a,b,c,d,e,f], that is the variable
        // "matrixEntries" is a row vector later on, this vector is mapped into the
        // Eigen matrix format
        std::vector<int> matrixEntries;

        // in this object we store the data from the matrix
        std::ifstream matrixDataFile(fileToOpen);
        if (!matrixDataFile.is_open()) {
            std::cerr << "Can not open the matrix in " << fileToOpen << std::endl;
        }

        // this variable is used to store the row of the matrix that contains commas
        std::string matrixRowString;

        // this variable is used to store the matrix entry;
        std::string matrixEntry;

        // this variable is used to track the number of rows
        int matrixRowNumber = 0;

        while (std::getline(
                matrixDataFile,
                matrixRowString)) // here we read a row by row of matrixDataFile and store
            // every line into the string variable matrixRowString
        {
            std::stringstream matrixRowStringStream(
                    matrixRowString); // convert matrixRowString that is a string to a
            // stream variable.

            while (
                    std::getline(matrixRowStringStream, matrixEntry,
                                 ',')) // here we read pieces of the stream
                // matrixRowStringStream until every comma, and store
                // the resulting character into the matrixEntry
            {
                matrixEntries.push_back(std::stoi(
                        matrixEntry)); // here we convert the string to double and fill in the
                // row vector storing all the matrix entries
            }
            matrixRowNumber++; // update the column numbers
        }

        // here we convet the vector variable into the matrix and return the resulting
        // object, note that matrixEntries.data() is the pointer to the first memory
        // location at which the entries of the vector matrixEntries are stored;
        return Eigen::Map<
                Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
                matrixEntries.data(), matrixRowNumber,
                matrixEntries.size() / matrixRowNumber);
    }

    void saveMat(std::string fileName,  /// <[in] output address
                 Eigen::MatrixXd matrix /// <[out] Dense matrix
    ) {
        // https://eigen.tuxfamily.org/dox/structEigen_1_1IOFormat.html
        const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision,
                                               Eigen::DontAlignCols, ", ", "\n");

        std::ofstream file(fileName);
        if (file.is_open()) {
            file << matrix.format(CSVFormat);
            file.close();
        }
    }

    bool sortbyfirst(const std::tuple<double, int> &a,
                     const std::tuple<double, int> &b) {
        return std::get<0>(a) < std::get<0>(b);
    }

    std::vector<int> getBiggestIds(std::vector<double> &input // The input array
    ) {
        std::vector<std::tuple<double, int>> tmp(input.size());
        for (int i = 0; i < input.size(); i++) {
            tmp[i] = std::tuple<double, int>(input[i], i);
        }
        std::sort(
                tmp.begin(), tmp.end(),
                [](const std::tuple<double, int> &a, const std::tuple<double, int> &b) {
                    return std::get<0>(a) > std::get<0>(b);
                });
        std::vector<int> index(tmp.size());
        for (int i = 0; i < tmp.size(); i++) {
            index[i] = std::get<1>(tmp[i]);
        }
        return index;
    }




///---------------------------------------------------------------------------------------\n
/// print_csc - save the csc sparse matrix in matrix market format (The matrix
/// is symmetric and lower triangular)
///---------------------------------------------------------------------------------------\n
    void print_csc(size_t n,   ///<[in] Number of nodes
                   int *Ap,    ///<[in] pointer array
                   int *Ai,    ///<[in] index array
                   double *Ax, ///<[in] value array
                   std::streambuf *out, const std::string indent,
                   const std::string &beg) {
        if (Ap == NULL || Ai == NULL) {
            std::cout << "The matrix is not defined" << std::endl;
            return;
        }
        std::streambuf *sb_cout_backup = std::cout.rdbuf();
        std::cout.rdbuf(out);
        std::cout << indent << beg << "\n";
        size_t nnz = n > 0 ? Ap[n] : 0;
        std::cout << indent << n << " " << n << " " << nnz << "\n";
        for (auto i = 0; i < n; ++i) {
            for (auto j = Ap[i]; j < Ap[i + 1]; ++j) {
                assert(j < nnz);
                if (Ax == NULL) {
                    std::cout << indent << Ai[j] + 1 << " " << i + 1 << " "
                              << std::setprecision(20) << 1;
                } else {
                    std::cout << indent << Ai[j] + 1 << " " << i + 1 << " "
                              << std::setprecision(20) << Ax[j];
                }

                std::cout << "\n";
            }
        }
        std::cout.rdbuf(sb_cout_backup);
    }


    bool isTetInSeg(std::vector<int> &Vert, std::vector<int> &group, int group_id) {
        if (group[Vert[0]] == group_id &&
            group[Vert[1]] == group_id &&
            group[Vert[2]] == group_id &&
            group[Vert[3]] == group_id) {
            return true;
        }
        return false;
    }

    void makeCounterClockwise(std::vector<int> &face, Eigen::MatrixXd &V) {
        Eigen::Vector3d v1 = V.row(face[1]) - V.row(face[0]);
        Eigen::Vector3d v2 = V.row(face[2]) - V.row(face[0]);
        if (v1.cross(v2).z() < 0) {
            std::swap(face[1], face[2]);
        }
    }

    bool WriteObjWithGroup(
            Eigen::MatrixXd V, ///[in] The nodal positions
            Eigen::MatrixXi F, /// [in] The faces
            std::vector<int> group, /// [in] a mapping from node to group
            std::vector<std::string> group_names, /// [in] A mapping from group to its name used for saving obj files
            std::string base_address /// The folder in which the obj files are saved
    ) {
        if (group.size() != V.rows()) {
            std::cerr << "The size of group mapping and nodes should be equal" << std::endl;
            return false;
        }

        std::vector<int> whole_to_seg_map(group.size(), -1);

        for (int group_id = 0; group_id < group_names.size(); group_id++) {
            std::ofstream obj_file;

            std::string file_name = base_address + "/" + group_names[group_id] + ".obj";
            obj_file.open(file_name, std::ios::out);

            if (!obj_file) {
                return false; // If file opening fails
            }

            int seg_v_cnt = 0;
            ///Writing Vertices
            for (int i = 0; i < V.rows(); i++) {
                if (group[i] == group_id) {
                    // Write vertices that belong to the current group
                    obj_file << "v " << V(i, 0) << " " << V(i, 1) << " " << V(i, 2) << "\n";
                    whole_to_seg_map[i] = seg_v_cnt;
                    seg_v_cnt++;
                }
            }

            ///Computing the SF from F
            Eigen::MatrixXi SF;
            std::vector<std::vector<int>> local_F_idx;
            for (int i = 0; i < F.rows(); i++) {
                std::vector<int> face1{F(i, 0), F(i, 1), F(i, 2)};
                std::sort(face1.begin(), face1.end());
                std::vector<int> face2{F(i, 0), F(i, 1), F(i, 3)};
                std::sort(face2.begin(), face2.end());
                std::vector<int> face3{F(i, 0), F(i, 2), F(i, 3)};
                std::sort(face3.begin(), face3.end());
                std::vector<int> face4{F(i, 1), F(i, 2), F(i, 3)};
                std::sort(face4.begin(), face4.end());
                std::vector<int> tet{F(i, 0), F(i, 1), F(i, 2), F(i, 3)};
                if (isTetInSeg(tet, group, group_id)) {
                    local_F_idx.emplace_back(tet);
                }
            }

            Eigen::MatrixXi local_F;
            local_F.resize(local_F_idx.size(), 4); // Resize local_F to appropriate size
            for (size_t i = 0; i < local_F_idx.size(); ++i) { // Iterate through local_F_idx
                // Convert std::vector to Eigen::Vector and assign to i-th row of local_F
                local_F.row(i) = Eigen::VectorXi::Map(local_F_idx[i].data(), local_F_idx[i].size());
            }
            igl::boundary_facets(local_F, SF);

            ///Writing Faces
            for (int i = 0; i < SF.rows(); i++) {
                if (group[SF(i, 0)] == group_id &&
                    group[SF(i, 1)] == group_id &&
                    group[SF(i, 2)] == group_id) {
                    if ((whole_to_seg_map[SF(i, 0)] == -1) ||
                        (whole_to_seg_map[SF(i, 1)] == -1) ||
                        (whole_to_seg_map[SF(i, 2)] == -1)) {
                        std::cerr << "The mapping from the whole obj to segmented obj is not correct" << std::endl;
                    }
                    // Write faces that have at least one vertex from the current group
                    obj_file << "f " << whole_to_seg_map[SF(i, 0)] + 1
                             << " " << whole_to_seg_map[SF(i, 1)] + 1
                             << " " << whole_to_seg_map[SF(i, 2)] + 1
                             << "\n"; // indices in OBJ file are 1-based
                }
            }

            obj_file.close();
        }
        return true;
    }


    ///---------------------------------------------------------------------------------------\n
    /// Write a single obj file that has U,V texture
    ///---------------------------------------------------------------------------------------\n
    bool applyTextureBasedOnLeavesAndSeparator(Eigen::MatrixXd V, ///[in] The nodal positions
                                               Eigen::MatrixXi SF, /// [in] Surface Faces
                                               int max_lvl, ///[in] the maximum lvl in the binary tree
                                               std::vector<int> group, /// [in] a mapping from node to group
                                               std::string base_address, /// The folder in which the obj files are saved
                                               int frame, int iter
    ) {
        if (group.size() != V.rows()) {
            std::cerr << "The size of group mapping and nodes should be equal" << std::endl;
            return false;
        }

        std::vector<double> binary_group(group.size(), 0);
        int start_idx_for_leaves = std::pow(2, max_lvl) - 1;
//        for(int i = binary_group.size() / 2; i < binary_group.size(); i++){
//            binary_group[i] = 1;
//        }
        for (int i = 0; i < group.size(); i++) {
            if (group[i] == -1) {
                binary_group[i] = 0.5;
            } else if (group[i] < start_idx_for_leaves) {
                binary_group[i] = 0;
            } else {
                binary_group[i] = 1;
            }
        }

        std::ofstream obj_file;

        std::string file_name =
                base_address + "/texture_" + std::to_string(frame) + "_" + std::to_string(iter) + ".obj";
        obj_file.open(file_name, std::ios::out);

        if (!obj_file) {
            return false; // If file opening fails
        }


        for (int i = 0; i < V.rows(); i++) {
            // Write vertices that belong to the current group
            obj_file << "v " << V(i, 0) << " " << V(i, 1) << " " << V(i, 2) << "\n";
        }

        // Write the texture vertices (u,v)
        for (int i = 0; i < binary_group.size(); i++) {
            if (binary_group[i] == 0) {
                obj_file << "vt " << "1 " << "1" << "\n";
            } else if (binary_group[i] == 1) {
                obj_file << "vt " << "0 " << "0" << "\n";
            } else {
                obj_file << "vt " << "0 " << "1" << "\n";
            }
        }

        for (int i = 0; i < SF.rows(); i++) {
            // Write faces that have at least one vertex from the current group
            obj_file << "f " << SF(i, 0) + 1 << "/" << SF(i, 0) + 1
                     << " " << SF(i, 1) + 1 << "/" << SF(i, 1) + 1
                     << " " << SF(i, 2) + 1 << "/" << SF(i, 2) + 1
                     << "\n"; // indices in OBJ file are 1-based
        }

        obj_file.close();


        return true;
    }
} // namespace PARTH
