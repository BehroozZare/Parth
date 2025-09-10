set(REQUIRED_CMAKE_VERSION "3.9")

################################################################################
# Dependencies
include(cli11)
include(eigen)
include(libigl)
include(polyscope)

################################################################################
# Create utils library
add_library(parth_dev_utils
    utils/vertex_colored_mesh.cpp
    utils/vertex_colored_mesh.h
    utils/polyscope_utils.cpp
    utils/polyscope_utils.h
)

target_include_directories(parth_dev_utils PUBLIC utils)
target_link_libraries(parth_dev_utils PUBLIC Eigen3::Eigen)
target_link_libraries(parth_dev_utils PUBLIC polyscope::polyscope)



# Testing Example Creation
add_executable(${PROJECT_NAME}_sep_runtime sep_runtime_analysis.cpp)
target_link_libraries(${PROJECT_NAME}_sep_runtime PUBLIC parth)
target_link_libraries(${PROJECT_NAME}_sep_runtime PUBLIC parth_dev_utils)
target_link_libraries(${PROJECT_NAME}_sep_runtime PUBLIC CLI11::CLI11)
target_link_libraries(${PROJECT_NAME}_sep_runtime PUBLIC Eigen3::Eigen)
target_link_libraries(${PROJECT_NAME}_sep_runtime PUBLIC igl::core)
target_link_libraries(${PROJECT_NAME}_sep_runtime PUBLIC polyscope::polyscope)

add_executable(${PROJECT_NAME}_see_separators see_separators.cpp)
target_link_libraries(${PROJECT_NAME}_see_separators PUBLIC parth)
target_link_libraries(${PROJECT_NAME}_see_separators PUBLIC parth_dev_utils)
target_link_libraries(${PROJECT_NAME}_see_separators PUBLIC CLI11::CLI11)
target_link_libraries(${PROJECT_NAME}_see_separators PUBLIC Eigen3::Eigen)
target_link_libraries(${PROJECT_NAME}_see_separators PUBLIC igl::core)
target_link_libraries(${PROJECT_NAME}_see_separators PUBLIC polyscope::polyscope)


add_executable(${PROJECT_NAME}_see_level_based_separators see_level_based_separators.cpp)
target_link_libraries(${PROJECT_NAME}_see_level_based_separators PUBLIC parth)
target_link_libraries(${PROJECT_NAME}_see_level_based_separators PUBLIC parth_dev_utils)
target_link_libraries(${PROJECT_NAME}_see_level_based_separators PUBLIC CLI11::CLI11)
target_link_libraries(${PROJECT_NAME}_see_level_based_separators PUBLIC Eigen3::Eigen)
target_link_libraries(${PROJECT_NAME}_see_level_based_separators PUBLIC igl::core)
target_link_libraries(${PROJECT_NAME}_see_level_based_separators PUBLIC polyscope::polyscope)


add_executable(${PROJECT_NAME}_see_patch_based_separators see_patch_based_separators.cpp)
target_link_libraries(${PROJECT_NAME}_see_patch_based_separators PUBLIC parth)
target_link_libraries(${PROJECT_NAME}_see_patch_based_separators PUBLIC parth_dev_utils)
target_link_libraries(${PROJECT_NAME}_see_patch_based_separators PUBLIC CLI11::CLI11)
target_link_libraries(${PROJECT_NAME}_see_patch_based_separators PUBLIC Eigen3::Eigen)
target_link_libraries(${PROJECT_NAME}_see_patch_based_separators PUBLIC igl::core)
target_link_libraries(${PROJECT_NAME}_see_patch_based_separators PUBLIC polyscope::polyscope)




# Polyscope example
add_executable(${PROJECT_NAME}_polyscope_example polyscope_example.cpp)
target_link_libraries(${PROJECT_NAME}_polyscope_example PUBLIC parth_dev_utils)
target_link_libraries(${PROJECT_NAME}_polyscope_example PUBLIC Eigen3::Eigen)
target_link_libraries(${PROJECT_NAME}_polyscope_example PUBLIC polyscope::polyscope)



