cmake_minimum_required(VERSION 3.14)

project(ParthAPIExamples)

# Find Parth package
find_package(parth REQUIRED)

# Create basic permutation demo
add_executable(basic_permutation_demo basic_permutation_demo.cpp)
target_link_libraries(basic_permutation_demo Parth::parth)

# Create dynamic mesh demo  
add_executable(dynamic_mesh_demo dynamic_mesh_demo.cpp)
target_link_libraries(dynamic_mesh_demo Parth::parth)

# Set C++ standard
set_target_properties(basic_permutation_demo dynamic_mesh_demo PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Add include for iomanip
target_compile_features(basic_permutation_demo PUBLIC cxx_std_17)
target_compile_features(dynamic_mesh_demo PUBLIC cxx_std_17)

message(STATUS "Parth API examples configured successfully")
