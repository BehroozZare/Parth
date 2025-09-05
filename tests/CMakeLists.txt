set(REQUIRED_CMAKE_VERSION "3.9")

################################################################################
# CLI11
include(cli11)

# Testing Example Creation
add_executable(${PROJECT_NAME}_example_creator example_creator.cpp)
target_link_libraries(${PROJECT_NAME}_example_creator PUBLIC parth)
target_link_libraries(${PROJECT_NAME}_example_creator PUBLIC CLI11::CLI11)

#include eigen
include(eigen)
target_link_libraries(${PROJECT_NAME}_example_creator PUBLIC Eigen3::Eigen)


#include libigl
include(libigl)
target_link_libraries(${PROJECT_NAME}_example_creator PUBLIC igl::core)