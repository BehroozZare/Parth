//
// Created by behrooz on 12/10/22.
//

#ifndef PARTH_SOLVER_NEWTONITERVISUALIZATION_H
#define PARTH_SOLVER_NEWTONITERVISUALIZATION_H

#include "Eigen/Eigen"
#include "GIF.h"
#include <igl/colormap.h>
#include <igl/opengl/glfw/Viewer.h>

namespace PARTH {

struct RGB {
    int R;
    int G;
    int B;
};

enum colorName { GREEN,
    BLUE,
    RED,
    WHITE,
    BROWN,
    YELLOW,
    PINK,
    PURPLE,
    ORANGE };

RGB getColor(colorName color);

///---------------------------------------------------------------------------------------\n
/// VisualizeNewtonIter - This class visualizes a frame's newton iterations
///---------------------------------------------------------------------------------------\n
class NewtonIterVisualization {
private:
    igl::opengl::glfw::Viewer viewer;
    double zoom;
    std::string sim_name;
    int view_num = 0;
    bool isInit;
    bool show_iter_positions = false;

    // Gif variables
    bool save_gif;
    GifWriter GIFWriter;
    uint32_t GIFDelay = 10; //*10ms
    int GIFStep = 1;
    double GIFScale = 0.6;
    std::string gif_address;

    // Classification Matrix
    Eigen::VectorXd group_vec;

    int num_nodes = 0;

    // Stack of position
    std::vector<Eigen::MatrixXd> iter_pos_stack;
    std::vector<Eigen::VectorXd> iter_groups_stack;
    std::vector<std::string> iter_name_stack;
    std::vector<std::tuple<int, int>> iter_frame_stack;

public:
    NewtonIterVisualization();

    ~NewtonIterVisualization() = default;

    void init(std::string sim_name, ///<[in] The name of the simulation
        bool save_gif, ///<[in] save the gif file
        std::string address ///<[in] address for saving the gif file
    );

    ///---------------------------------------------------------------------------------------\n
    /// addIterSol - save data for later visualization in the cache \n
    ///---------------------------------------------------------------------------------------\n
    void addIterSol(const Eigen::MatrixXd& sol, const double* group_vec,
        int frame_num, int iter_num, std::string name = "");

    void addIterSol(const Eigen::MatrixXd& sol, const int* group_vec,
        int frame_num, int iter_num, std::string name = "");

    ///---------------------------------------------------------------------------------------\n
    /// showErrorChanges - This function visual the change in error in a
    /// single frame
    ///---------------------------------------------------------------------------------------\n
    void
    visualizeNewtonIters(const Eigen::MatrixXd& V_rest, ///<[in] resting position
        const Eigen::MatrixXi& SF, ///<[in] surface mesh
        int num_groups);

    ///---------------------------------------------------------------------------------------\n
    /// initGif - Init gif writer characteristics\n
    ///---------------------------------------------------------------------------------------\n
    void initGif();

    ///---------------------------------------------------------------------------------------\n
    /// writeGif - Writing to a gif file\n
    ///---------------------------------------------------------------------------------------\n
    void writeGif();
};
} // namespace PARTH

#endif // PARTH_SOLVER_NEWTONITERVISUALIZATION_H
