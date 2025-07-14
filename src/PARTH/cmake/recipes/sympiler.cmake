# Eigen Gurobi Wrapper
if (TARGET Sym::EigenGurobi)
    return()
endif ()

message(STATUS "Third-party: creating target 'SympilerEigen::SympilerEigen'")

include(FetchContent)
FetchContent_Declare(
        sympiler_eigen
        GIT_REPOSITORY https://github.com/BehroozZare/sympiler-eigen.git
        GIT_TAG 8ee49488d72bdd253389d22d282d1d4ca59fc9a3
)


FetchContent_MakeAvailable(sympiler_eigen)
add_library(SympilerEigen::SympilerEigen ALIAS sympiler_eigen)

FetchContent_GetProperties(sympiler_eigen)
if (NOT sympiler_eigen_POPULATED)
    FetchContent_Populate(sympiler_eigen)
endif ()
set(SYMPILER_EIGEN_INCLUDE_DIRS ${sympiler_eigen_SOURCE_DIR}/)


set(SYMPILER_INCLUDE_DIRS
        ${SYMPILER_EIGEN_INCLUDE_DIRS}/includes
        ${SYMPILER_EIGEN_INCLUDE_DIRS}/sympiler/includes/
        ${CMAKE_CURRENT_SOURCE_DIR}/sympiler/lbc/includes/
        ${EIGEN_INCLUDE_DIRS}
        )

set_target_properties(sympiler_eigen PROPERTIES EXCLUDE_FROM_ALL TRUE)
