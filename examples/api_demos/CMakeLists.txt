set(REQUIRED_CMAKE_VERSION "3.9")

################################################################################
# CLI11
include(cli11)

# Testing Permutation kernel
add_executable(${PROJECT_NAME}_perm permutation.cpp)
target_link_libraries(${PROJECT_NAME}_perm PUBLIC parth)
target_link_libraries(${PROJECT_NAME}_perm PUBLIC CLI11::CLI11)