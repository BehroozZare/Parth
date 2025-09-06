# Check required CMake version
set(REQUIRED_CMAKE_VERSION "3.14")
set(CMAKE_POLICY_DEFAULT_CMP0127 NEW)
cmake_minimum_required(VERSION ${REQUIRED_CMAKE_VERSION})

project(PARTH DESCRIPTION "A development code base" LANGUAGES CXX)

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I/usr/local/opt/include -L/usr/local/lib")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/usr/local/opt/include -L/usr/local/lib/c++")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)


################################################################################
# Test options
option(${PROJECT_NAME}_WITH_TESTS "Add tests" ON)

#Add solvers wrapper
option(${PROJECT_NAME}_WITH_SOLVER_WRAPPER "Add solver wrapper" OFF)


#Examples options
option(${PROJECT_NAME}_WITH_API_DEMO "Add permutation demo" ON)
option(${PROJECT_NAME}_WITH_CHOLMOD_DEMO "Add cholmod demo" OFF)
option(${PROJECT_NAME}_WITH_ACCELERATE_DEMO "Add accelerate demo" OFF)
option(${PROJECT_NAME}_WITH_MKL_DEMO "Add mkl demo" OFF)
option(${PROJECT_NAME}_WITH_IPC_DEMO "Add strumpack demo" OFF)
option(${PROJECT_NAME}_WITH_REMESHING_DEMO "Add remeshing demo" OFF)
option(${PROJECT_NAME}_WITH_SOLVER_WRAPPER_DEMO "Add solver wrapper demo" OFF)


if (APPLE)
    set(${PROJECT_NAME}_WITH_MKL OFF)
endif ()
################################################################################

### Configuration
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/find)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/recipes)

################################################################################
# SRCS Library
################################################################################

# Core Parth library source files
file(GLOB PARTH_CORE_SRCFILES
        "src/Parth/*.cpp"
)

# Create the main Parth library (core functionality)
add_library(parth ${PARTH_CORE_SRCFILES})
add_library(Parth::parth ALIAS parth)

# Set target properties for proper library export
set_target_properties(parth PROPERTIES
    VERSION 1.0.0
    SO_VERSION 1
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(parth PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/Parth/include>
    $<INSTALL_INTERFACE:include>
)

target_include_directories(parth PUBLIC
        "src/Parth/include"
)

################################################################################
# Compiler options
################################################################################
add_compile_options(-Wno-write-strings)
add_compile_options(-Wunused-result)

#find_package(OPENMP)

# Figure out SSE level support
message(STATUS "Seaching for SSE...")
find_package(SSE)

# Figure out AVX level support
message(STATUS "Searching for AVX...")
find_package(AVX)

# Figure out FMA level support
message(STATUS "Searching for FMA...")
find_package(FMA)

# Add SSE, AVX, and FMA flags to compiler flags
string(REPLACE " " ";" SIMD_FLAGS "${SSE_FLAGS} ${AVX_FLAGS} ${FMA_FLAGS}")
target_compile_options(parth PUBLIC ${SIMD_FLAGS})

# Add -pthread to compilation and linking
find_package(Threads)
target_link_libraries(parth PUBLIC ${CMAKE_THREAD_LIBS_INIT})

# Use C++11
target_compile_features(parth PUBLIC cxx_std_17)

################################################################################
# Required Libraries
################################################################################
# Find Metis
find_package(METIS REQUIRED)
if (EXISTS "${METIS_INCLUDES}")
    target_link_libraries(parth PUBLIC
            ${METIS_LIBRARIES}
    )
    target_link_libraries(parth PUBLIC ${METIS_LIBRARIES})
    target_include_directories(parth PUBLIC "${METIS_INCLUDES}")
else ()
    message(STATUS "The METIS is not found")
endif ()

# Find SuiteSparse
set(SUITESPARSE_INCLUDE_DIR_HINTS $ENV{SUITESPARSE_INC})
set(SUITESPARSE_LIBRARY_DIR_HINTS $ENV{SUITESPARSE_LIB})
find_package(SuiteSparse REQUIRED)
target_link_libraries(parth PUBLIC ${SUITESPARSE_LIBRARIES})
target_include_directories(parth PUBLIC ${SUITESPARSE_INCLUDE_DIRS})

# Find Accelerate Framework
if (APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if (ACCELERATE_FRAMEWORK)
        message(STATUS "Accelerate Framework found: ${ACCELERATE_FRAMEWORK}")
        target_link_libraries(parth PUBLIC ${ACCELERATE_FRAMEWORK})
        target_compile_definitions(parth PUBLIC PARTH_WITH_ACCELERATE)
    else ()
        message(FATAL_ERROR "Accelerate Framework not found")
    endif ()
endif ()

#Strumpack
if (${PROJECT_NAME}_WITH_STRUMPACK)
    message(STATUS "FINDING STRUMPACK")

    set(STRUMPACK_DIR "$ENV{STRUMPACKROOT}/lib/cmake/STRUMPACK/")
    find_package(STRUMPACK REQUIRED)
    target_link_libraries(parth PRIVATE STRUMPACK::strumpack)
endif ()


find_package(OpenMP)
if (OPENMP_FOUND)
    #include_directories("${OPENMP_INCLUDES}")
    #include_directories("/usr/local/include" "/usr/local/opt/llvm/include")
    #link_directories("${OPENMP_LIBRARIES}")
    #link_directories("/usr/local/lib" "/usr/local/opt/llvm/lib")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    add_definitions(-DOPENMP)
endif ()

if (APPLE)
    set(OMP_BREW /opt/homebrew/Cellar/libomp/21.1.0/)
    target_include_directories(parth PUBLIC ${OMP_BREW}/include)
    find_library(OMP NAMES omp HINTS ${OMP_BREW}/lib)
    if (OMP)
        message(STATUS "Found omp: ${OMP}")
    else ()
        message(FATAL_ERROR "omp library not found")
    endif ()
    target_link_libraries(parth PUBLIC ${OMP})
endif ()

# ################################################################################
# # tests
# ################################################################################
if (${PROJECT_NAME}_WITH_TESTS)
    add_subdirectory(tests)
endif ()

# ################################################################################
# # examples
# ################################################################################
if (${PROJECT_NAME}_WITH_API_DEMO)
    add_subdirectory(examples)
endif ()

if (${PROJECT_NAME}_WITH_CHOLMOD_DEMO)
    add_subdirectory(examples/api_demos/cholmod_computation)
endif ()

if (${PROJECT_NAME}_WITH_ACCELERATE_DEMO)
    add_subdirectory(examples/api_demos/accelerate_computation)
endif ()

if (${PROJECT_NAME}_WITH_MKL_DEMO)
    add_subdirectory(examples/api_demos/mkl_computation)
endif ()

if (${PROJECT_NAME}_WITH_IPC_DEMO)
    add_subdirectory(examples/api_demos/ipc_computation)
endif ()

if (${PROJECT_NAME}_WITH_REMESHING_DEMO)
    add_subdirectory(examples/api_demos/remeshing_computation)
endif ()

# ################################################################################
# # Install targets and configuration
# ################################################################################

# # Install targets and configuration
# include(GNUInstallDirs)

# # Install the library
# install(TARGETS parth
#     EXPORT parthTargets
#     LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
# )

# # Install public headers
# install(FILES include/parth/parth.h
#     DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/parth
# )

# # Install internal headers needed by the API
# install(DIRECTORY src/Parth/include/
#     DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/parth/internal
#     FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
# )

# # Generate and install CMake config files
# include(CMakePackageConfigHelpers)

# # Generate the config file
# configure_package_config_file(
#     "${CMAKE_CURRENT_SOURCE_DIR}/cmake/parthConfig.cmake.in"
#     "${CMAKE_CURRENT_BINARY_DIR}/parthConfig.cmake"
#     INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/parth
# )

# # Generate the version file
# write_basic_package_version_file(
#     "${CMAKE_CURRENT_BINARY_DIR}/parthConfigVersion.cmake"
#     VERSION "1.0.0"
#     COMPATIBILITY AnyNewerVersion
# )

# # Install the config files
# install(FILES
#     "${CMAKE_CURRENT_BINARY_DIR}/parthConfig.cmake"
#     "${CMAKE_CURRENT_BINARY_DIR}/parthConfigVersion.cmake"
#     DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/parth
# )

# # Install the targets file
# install(EXPORT parthTargets
#     FILE parthTargets.cmake
#     NAMESPACE Parth::
#     DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/parth
# )


