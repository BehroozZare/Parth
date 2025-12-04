# SuiteSparse AMD recipe: find installed or download and build
#
# This recipe first tries to find an installed AMD library.
# If not found, it downloads and builds AMD from source.
# AMD does not require BLAS/LAPACK.

if(TARGET SuiteSparse::AMD)
    return()
endif()

# First, try to find an installed SuiteSparse/AMD
find_package(SuiteSparse QUIET)

if(SUITESPARSE_FOUND AND AMD_LIBRARY)
    message(STATUS "Found installed SuiteSparse AMD: ${AMD_LIBRARY}")
    
    if(NOT TARGET SuiteSparse::AMD)
        add_library(SuiteSparse::AMD INTERFACE IMPORTED)
        set_target_properties(SuiteSparse::AMD PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${AMD_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES "${AMD_LIBRARY};${SUITESPARSE_CONFIG_LIBRARY}"
        )
    endif()
else()
    message(STATUS "SuiteSparse AMD not found, downloading and building from source...")
    
    # Configure SuiteSparse build options - AMD only (no BLAS/LAPACK needed)
    set(SUITESPARSE_ENABLE_PROJECTS "suitesparse_config;amd" CACHE STRING "SuiteSparse projects to build")
    
    # Disable demo/test builds
    set(SUITESPARSE_DEMOS OFF CACHE BOOL "Disable demos")
    
    # Disable Fortran
    set(SUITESPARSE_USE_FORTRAN OFF CACHE BOOL "Disable Fortran")
    
    # Enable building static libraries
    set(BUILD_STATIC_LIBS ON CACHE BOOL "Build static libraries")
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Don't build shared libraries")
    
    include(FetchContent)
    FetchContent_Declare(
        suitesparse
        GIT_REPOSITORY https://github.com/DrTimothyAldenDavis/SuiteSparse.git
        GIT_TAG v7.11.0
        GIT_SHALLOW TRUE
        EXCLUDE_FROM_ALL
    )
    
    message(STATUS "Fetching SuiteSparse (AMD only)...")
    FetchContent_MakeAvailable(suitesparse)
    message(STATUS "SuiteSparse configured with AMD only")
    
    # Create alias targets for consistent naming
    if(NOT TARGET SuiteSparse::AMD)
        if(TARGET AMD_static)
            add_library(SuiteSparse::AMD ALIAS AMD_static)
        elseif(TARGET AMD)
            add_library(SuiteSparse::AMD ALIAS AMD)
        endif()
    endif()
    
    if(NOT TARGET SuiteSparse::SuiteSparseConfig)
        if(TARGET SuiteSparseConfig_static)
            add_library(SuiteSparse::SuiteSparseConfig ALIAS SuiteSparseConfig_static)
        elseif(TARGET SuiteSparseConfig)
            add_library(SuiteSparse::SuiteSparseConfig ALIAS SuiteSparseConfig)
        endif()
    endif()
    
    # Export include directories
    set(SUITESPARSE_INCLUDE_DIRS 
        ${suitesparse_SOURCE_DIR}/AMD/Include
        ${suitesparse_SOURCE_DIR}/SuiteSparse_config
        CACHE PATH "SuiteSparse include directories"
    )
    
    message(STATUS "SuiteSparse AMD will be built from source: ${suitesparse_SOURCE_DIR}")
endif()
