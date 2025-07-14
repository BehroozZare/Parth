# Check required CMake version
set(REQUIRED_CMAKE_VERSION "3.14")
set(CMAKE_POLICY_DEFAULT_CMP0127 NEW)
cmake_minimum_required(VERSION ${REQUIRED_CMAKE_VERSION})

project(PARTH_SOLVER DESCRIPTION "A development code base" LANGUAGES CXX)

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I/usr/local/opt/include -L/usr/local/lib")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/usr/local/opt/include -L/usr/local/lib/c++")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)


################################################################################

#Project options
option(${PROJECT_NAME}_WITH_SPMP "Use SpMP library with HDAGG" OFF)
option(${PROJECT_NAME}_WITH_DAGP "Add DAGP for benchmarking" OFF)
option(${PROJECT_NAME}_WITH_CHOLMOD "Add Cholmod for benchmarking" ON)
option(${PROJECT_NAME}_WITH_STRUMPACK "Add Strumpack for benchmarking" OFF)
option(${PROJECT_NAME}_WITH_MKL "Add MKL BLAS" ON)
option(${PROJECT_NAME}_WITH_ACCELERATE "Add MKL BLAS" ON)
option(${PROJECT_NAME}_WITH_METIS "Add METIS Ordering" ON)
option(${PROJECT_NAME}_WITH_OPENGL "Add OPENGL Ordering" ON)
option(${PROJECT_NAME}_WITH_SYMPILER "Add OPENGL Ordering" OFF)

option(${PROJECT_NAME}_WITH_PARTH "Add PARTH Ordering" ON)
option(${PROJECT_NAME}_WITH_DEMO "Add demo executables" ON)

if (APPLE)
    set(${PROJECT_NAME}_WITH_MKL OFF)
endif ()
################################################################################

### Configuration
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/find)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/recipes)

################################################################################
# LBC Library
################################################################################

# project source files
file(GLOB SRCFILES
        "src/LinSysSolver/*.cpp"
        "src/remesher/*.cpp"
        "src/DemoUtils/*.cpp"
        "src/ParthRefactor/*.cpp"
        "src/Barb/*.cpp"
        "src/Barb/CustomCholmod/*/*.cpp"
#        "src/SpMP/*.cpp"
#        "src/SpMP/synk/*.cpp"
#        "src/SpMP/reordering/*.cpp"
)

add_library(${PROJECT_NAME}_lib ${SRCFILES})

target_include_directories(${PROJECT_NAME}_lib PUBLIC
        "src/LinSysSolver"
        "src/remesher"
        "src/DemoUtils/include"
        "src/ParthRefactor/include"
        "src/ParthRefactor/include"
        "src/Barb/CustomCholmod/Cholmod_Include"
        "src/Barb/CustomCholmod/include"
        "src/Barb"
#        "src/SpMP"
#        "src/SpMP/synk"
#        "src/SpMP/reordering"
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
target_compile_options(${PROJECT_NAME}_lib PUBLIC ${SIMD_FLAGS})

# Add -pthread to compilation and linking
find_package(Threads)
target_link_libraries(${PROJECT_NAME}_lib PUBLIC ${CMAKE_THREAD_LIBS_INIT})

# Use C++11
target_compile_features(${PROJECT_NAME}_lib PUBLIC cxx_std_17)

################################################################################
# Required Libraries
################################################################################

# CLI11
include(cli11)
target_link_libraries(${PROJECT_NAME}_lib PUBLIC CLI11::CLI11)

### Find Metis

if (${PROJECT_NAME}_WITH_METIS)
    find_package(METIS REQUIRED)
    if (EXISTS "${METIS_INCLUDES}")
        target_link_libraries(${PROJECT_NAME}_lib PUBLIC
                ${METIS_LIBRARIES}
        )
        target_include_directories(${PROJECT_NAME}_lib PUBLIC "${METIS_INCLUDES}")
    else ()
        message(STATUS "The METIS is not found")
    endif ()
endif ()

### Find Intel MKL if available
if (${PROJECT_NAME}_WITH_MKL)
    find_package(MKL OPTIONAL_COMPONENTS)
    if (EXISTS "${MKL_INCLUDE_DIRS}")
        message(STATUS "The MKL has been found")
        target_include_directories(${PROJECT_NAME}_lib PUBLIC "${MKL_INCLUDE_DIRS}")
        target_link_libraries(${PROJECT_NAME}_lib PUBLIC ${MKL_LIBRARIES})
        target_compile_definitions(${PROJECT_NAME}_lib PUBLIC PARTH_WITH_MKL)
    else ()
        message(STATUS "Intel MKL not found.")
        set(MKL_LIBRARIES "")
        set(MKL 0)
    endif ()

endif ()


### Check whether DAGP root exists - Relative works
if (${PROJECT_NAME}_WITH_DAGP)
    if (EXISTS "$ENV{DAGPROOT}")
        message(STATUS "The DAGP is found - $ENV{DAGPROOT}.")
        target_include_directories(${PROJECT_NAME}_lib PUBLIC
                "$ENV{DAGPROOT}/src/recBisection/"
                "$ENV{DAGPROOT}/src/common/"
                "src/utils/external/DAGPInterface/includes"
        )
        target_link_libraries(${PROJECT_NAME}_lib PUBLIC
                "$ENV{DAGPROOT}/lib/libdagp.a"
                "src/utils/external/DAGPInterface/*.cpp"
        )
        add_definitions(-DDAGP)
    else ()
        message(STATUS "Please defined the DAGP directory using DAGP_ROOT variable")
    endif ()
endif ()

### Check whether SPMP root exists - Relative works
if (${PROJECT_NAME}_WITH_SPMP)
    if (EXISTS "$ENV{SPMPROOT}")
        message(STATUS "Found SpMP - $ENV{SPMPROOT}.")
        target_include_directories(${PROJECT_NAME}_lib PUBLIC
                $ENV{SPMPROOT}
                "src/utils/external/SpMPInterface/includes"
        )
        target_link_libraries(${PROJECT_NAME}_lib PUBLIC
                "$ENV{SPMPROOT}/libspmp.a"
        )
        add_definitions(-DSPMP)
    else ()
        message(STATUS "Please defined the SpMP directory using SPMPROOT variable - not found")
    endif ()
endif ()

# SuiteSparse
if (${PROJECT_NAME}_WITH_CHOLMOD)
    set(SUITESPARSE_INCLUDE_DIR_HINTS $ENV{SUITESPARSE_INC})
    set(SUITESPARSE_LIBRARY_DIR_HINTS $ENV{SUITESPARSE_LIB})
    find_package(SuiteSparse REQUIRED)
    target_link_libraries(${PROJECT_NAME}_lib PUBLIC ${SUITESPARSE_LIBRARIES})
    target_include_directories(${PROJECT_NAME}_lib PUBLIC ${SUITESPARSE_INCLUDE_DIRS})
    target_compile_definitions(${PROJECT_NAME}_lib PUBLIC PARTH_WITH_CHOLMOD)
endif ()


#Strumpack
if (${PROJECT_NAME}_WITH_STRUMPACK)
    message(STATUS "FINDING STRUMPACK")

    set(STRUMPACK_DIR "$ENV{STRUMPACKROOT}/lib/cmake/STRUMPACK/")
    find_package(STRUMPACK REQUIRED)
    target_include_directories(${PROJECT_NAME}_lib PUBLIC "$ENV{STRUMPACKROOT}/include")
    target_link_libraries(${PROJECT_NAME}_lib PRIVATE STRUMPACK::strumpack)
endif ()

# Find Eigen
include(eigen)
target_link_libraries(${PROJECT_NAME}_lib PUBLIC Eigen3::Eigen)


# libigl
include(libigl)
# Other modules you could enable
igl_include(embree)
igl_include(imgui)
#igl_include(opengl)
#igl_include(stb)
#igl_include(predicates)
#igl_include(xml)
#igl_include(copyleft cgal)
#igl_include(copyleft comiso)
igl_include(copyleft core)
#igl_include(copyleft cork)
igl_include(copyleft tetgen)
#igl_include(restricted matlab)
#igl_include(restricted mosek)
igl_include(restricted triangle)


target_link_libraries(${PROJECT_NAME}_lib PUBLIC igl_copyleft::core igl_restricted::triangle igl_copyleft::tetgen)
if (${PROJECT_NAME}_WITH_OPENGL)
    igl_include(glfw)
    target_link_libraries(${PROJECT_NAME}_lib PUBLIC igl::glfw igl::imgui igl::embree)
    target_compile_definitions(${PROJECT_NAME}_lib PUBLIC USE_OPENGL)
endif ()

# Find Accelerate Framework
if (APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if (ACCELERATE_FRAMEWORK)
        message(STATUS "Accelerate Framework found: ${ACCELERATE_FRAMEWORK}")
        target_link_libraries(${PROJECT_NAME}_lib PUBLIC ${ACCELERATE_FRAMEWORK})
        target_compile_definitions(${PROJECT_NAME}_lib PUBLIC PARTH_WITH_ACCELERATE)
    else ()
        message(FATAL_ERROR "Accelerate Framework not found")
    endif ()
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
    set(OMP_BREW /usr/local/Cellar/libomp/18.1.4/)
    target_include_directories(${PROJECT_NAME}_lib PUBLIC ${OMP_BREW}/include)
    find_library(OMP NAMES omp HINTS ${OMP_BREW}/lib)
    if (OMP)
        message(STATUS "Found omp: ${OMP}")
    else ()
        message(FATAL_ERROR "omp library not found")
    endif ()
    target_link_libraries(${PROJECT_NAME}_lib PUBLIC ${OMP})
endif ()

if (${PROJECT_NAME}_WITH_SYMPILER)
    message(STATUS "ADDING SYMPILER")
    include(sympiler)

    target_link_libraries(${PROJECT_NAME}_lib PUBLIC SympilerEigen::SympilerEigen)
    # TODO: make it only one header
    message(STATUS "THE SYMPILER INCLUDE DIR IS")
    message(STATUS ${SYMPILER_INCLUDE_DIRS})
    target_include_directories(${PROJECT_NAME}_lib PUBLIC ${SYMPILER_INCLUDE_DIRS}
    )
    target_compile_definitions(${PROJECT_NAME}_lib PUBLIC PARTH_WITH_SYMPILER)
endif ()


# Develop folder
if (${PROJECT_NAME}_WITH_DEMO)
    add_subdirectory(demo)
endif ()


