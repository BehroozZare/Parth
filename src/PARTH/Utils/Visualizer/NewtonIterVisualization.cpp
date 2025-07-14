//
// Created by behrooz on 12/10/22.
//

#include "NewtonIterVisualization.h"

namespace PARTH {
//========================== VisualizeNewton Class ============================
NewtonIterVisualization::NewtonIterVisualization() { isInit = false; }

void NewtonIterVisualization::init(std::string sim_name, bool save_gif,
    std::string address)
{
    if (isInit) {
        return;
    }
    isInit = true;

    // giff stuff
    this->save_gif = save_gif;
    this->gif_address = address;
    this->sim_name = sim_name;

    GIFDelay = 25; //*10ms
    GIFStep = 1;
    GIFScale = 0.6;

    // Setup viewer
    zoom = 1.0;
    viewer.core().background_color << 1.0f, 1.0f, 1.0f, 0.0f;
    viewer.data().show_lines = true;
    viewer.core().lighting_factor = 1.0;
    viewer.data().show_texture = true;
    viewer.core().orthographic = false;
    viewer.core().camera_zoom *= zoom;
    viewer.core().animation_max_fps = 2.0;
    viewer.data().point_size = 5;
    viewer.data().show_overlay = true;
    viewer.core().trackball_angle = Eigen::Quaternionf(
        Eigen::AngleAxisf(M_PI_4 / 2.0, Eigen::Vector3f::UnitX()));

    group_vec.setOnes();
}

void NewtonIterVisualization::addIterSol(const Eigen::MatrixXd& sol,
    const double* group_vec, int frame_num,
    int iter_num, std::string name)
{
    this->iter_pos_stack.emplace_back(sol);
    this->num_nodes = sol.rows();
    Eigen::VectorXd tmp(num_nodes);
    for (int i = 0; i < num_nodes; i++) {
        tmp[i] = group_vec[i];
    }
    this->iter_groups_stack.emplace_back(tmp);
    this->iter_frame_stack.emplace_back(
        std::tuple<int, int>(frame_num, iter_num));
    this->iter_name_stack.emplace_back(name);
}

void NewtonIterVisualization::addIterSol(const Eigen::MatrixXd& sol,
    const int* group_vec, int frame_num,
    int iter_num, std::string name)
{
    this->iter_pos_stack.emplace_back(sol);
    this->num_nodes = sol.rows();
    Eigen::VectorXd tmp(num_nodes);
    for (int i = 0; i < num_nodes; i++) {
        tmp[i] = group_vec[i];
    }
    this->iter_groups_stack.emplace_back(tmp);
    this->iter_frame_stack.emplace_back(
        std::tuple<int, int>(frame_num, iter_num));
    this->iter_name_stack.emplace_back(name);
}

void NewtonIterVisualization::visualizeNewtonIters(
    const Eigen::MatrixXd& V_rest, const Eigen::MatrixXi& SF, int num_groups)
{

    bool init_sim = true;
    bool init_gif = false;
    int current_iter = 0;
    this->view_num = 0;
    Eigen::MatrixXd group_color;

    auto oneStepForwad = [&]() {
        viewer.core().lighting_factor = 0.6;
        viewer.core().align_camera_center(V_rest, SF);
        std::cout << "Camera: " << viewer.core().camera_view_angle << std::endl;
        std::cout << "Iter name: " << iter_name_stack[current_iter] << " Progress: "
                  << ((current_iter * 1.0) / iter_pos_stack.size()) * 100
                  << std::endl;
        viewer.data().show_texture = false;
        viewer.data().set_mesh(iter_pos_stack[current_iter], SF);
        viewer.data().set_uv(iter_pos_stack[current_iter]);

        Eigen::VectorXd group_vec = iter_groups_stack[current_iter];

        if (num_groups > 8) {
            group_color = (group_vec.rowwise() - group_vec.colwise().minCoeff())
                              .array()
                              .rowwise()
                / (group_vec.colwise().maxCoeff() - group_vec.colwise().minCoeff())
                      .array();
            //      Eigen::VectorXd zeros(group_vec.size());
            //      zeros.setZero();
            //      Eigen::VectorXd ones(group_vec.size());
            //      ones.setOnes();
            //      group_color =
            //          (group_vec.rowwise() -
            //          zeros.colwise().minCoeff()).array().rowwise() /
            //          (ones.colwise().maxCoeff() -
            //          zeros.colwise().minCoeff()).array();
            viewer.data().set_data(group_color);
        }
        else {
            group_color.resize(group_vec.size(), 3);
            for (int i = 0; i < group_vec.size(); i++) {
                RGB elem_color;
                if (group_vec[i] == 0) {
                    elem_color = getColor(colorName::GREEN);
                }
                else if (group_vec[i] == 1) {
                    elem_color = getColor(colorName::RED);
                }
                else if (group_vec[i] == 2) {
                    elem_color = getColor(colorName::BLUE);
                }
                else if (group_vec[i] == 3) {
                    elem_color = getColor(colorName::ORANGE);
                }
                else if (group_vec[i] == 4) {
                    elem_color = getColor(colorName::PURPLE);
                }
                else if (group_vec[i] == 5) {
                    elem_color = getColor(colorName::BROWN);
                }
                else if (group_vec[i] == 6) {
                    elem_color = getColor(colorName::YELLOW);
                }
                else if (group_vec[i] == 7) {
                    elem_color = getColor(colorName::PINK);
                }
                else {
                    std::cerr << "The number of groups are more than 3 - group vector "
                                 "value is: "
                              << group_vec[i] << std::endl;
                }
                group_color(i, 0) = elem_color.R;
                group_color(i, 1) = elem_color.G;
                group_color(i, 2) = elem_color.B;
                viewer.data().set_colors(group_color);
            }
        }
    };

    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer& viewer) -> bool {
        if (viewer.core().is_animating) {
            oneStepForwad();
            if (init_sim) {
                init_sim = false;
            }
            else {
                current_iter++;
            }
        }
        return false;
    };

    viewer.callback_post_draw = [&](igl::opengl::glfw::Viewer& viewer) -> bool {
        if (viewer.core().is_animating) {

            if (current_iter > 0 && save_gif && !init_sim) {
                printf("The sizes are: %.0f, %.0f, %.0f, %.0f\n",
                    viewer.core().viewport[0], viewer.core().viewport[1],
                    viewer.core().viewport[2], viewer.core().viewport[3]);

                if (current_iter == 1 || !init_gif) {
                    initGif();
                    init_gif = true;
                }
                writeGif();
            }

            if (current_iter == iter_pos_stack.size()) {
                GifEnd(&GIFWriter);
                current_iter = 0;
                init_sim = true;
                viewer.data().clear();
                viewer.core().is_animating = true;
                viewer.callback_pre_draw(viewer);
                viewer.callback_post_draw(viewer);
                viewer.core().is_animating = false;
                view_num++;
            }
        }

        return false;
    };

    viewer.callback_key_pressed = [&](igl::opengl::glfw::Viewer&,
                                      unsigned int key, int mod) {
        switch (key) {
        default:
            return false;
        case ' ': // Pause and Start
            viewer.core().is_animating = !viewer.core().is_animating;
            return true;
        case 'e': // Shutting down the animation
            viewer.launch_shut();
            return true;
        case 'p': // Activate changes in the position of the sphere
            std::cout << "The changes in position will be shown" << std::endl;
            this->show_iter_positions = true;
            return true;
        case 'r': // Resetting the simulation
            std::cout << "Resetting..." << std::endl;
            current_iter = 0;
            return true;
        case 's': // Save the animation
            GifEnd(&GIFWriter);
            current_iter = 0;
            init_gif = false;
            view_num++;
            return true;
        case 'f': // One step forward
            oneStepForwad();
            if (init_sim) {
                init_sim = false;
            }
            else {
                current_iter++;
            }
            if (!init_gif) {
                initGif();
                init_gif = true;
            }
            writeGif();
            if (current_iter == iter_pos_stack.size()) {
                current_iter = 0;
            }
            return true;
        case 'b': // One step back
            if (current_iter > 0) {
                current_iter--;
            }
            oneStepForwad();
            return true;
        }
    };

    viewer.core().is_animating = true;
    viewer.callback_pre_draw(viewer);
    viewer.callback_post_draw(viewer);
    viewer.core().is_animating = false;
    viewer.launch();
}

// GIF Wrapper
void NewtonIterVisualization::initGif()
{
    GifBegin(&GIFWriter,
        (gif_address + sim_name + "_" + std::to_string(view_num) + ".gif")
            .c_str(),
        (viewer.core().viewport[2] - viewer.core().viewport[0]),
        (viewer.core().viewport[3] - viewer.core().viewport[1]), GIFDelay);
}

void NewtonIterVisualization::writeGif()
{
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

RGB getColor(colorName color)
{
    RGB out{};
    switch (color) {
    case colorName::RED:
        out.R = 255;
        out.G = 0;
        out.B = 0;
        return out;
    case colorName::BLUE:
        out.R = 0;
        out.G = 0;
        out.B = 255;
        return out;
    case colorName::GREEN:
        out.R = 0;
        out.G = 255;
        out.B = 0;
        return out;
    case colorName::ORANGE:
        out.R = 255;
        out.G = 165;
        out.B = 0;
        return out;
    case colorName::BROWN:
        out.R = 150;
        out.G = 75;
        out.B = 0;
        return out;
    case colorName::YELLOW:
        out.R = 255;
        out.G = 215;
        out.B = 0;
        return out;
    case colorName::PURPLE:
        out.R = 221;
        out.G = 160;
        out.B = 221;
        return out;
    case colorName::PINK:
        out.R = 255;
        out.G = 105;
        out.B = 180;
        return out;
    case colorName::WHITE:
        out.R = 255;
        out.G = 255;
        out.B = 255;
        return out;
    default:
        out.R = 0;
        out.G = 0;
        out.B = 0;
        return out;
    }
}

} // namespace PARTH