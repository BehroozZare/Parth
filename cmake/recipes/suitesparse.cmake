# SuiteSparse recipe: find installed or download and build
#
# This recipe first tries to find an installed SuiteSparse library.
# If not found, it downloads and builds SuiteSparse from source.
# Only CHOLMOD and AMD (with dependencies) are built.

if(TARGET SuiteSparse::CHOLMOD)
    return()
endif()

# First, try to find an installed SuiteSparse
find_package(SuiteSparse QUIET)

if(SUITESPARSE_FOUND AND CHOLMOD_LIBRARY AND AMD_LIBRARY)
    message(STATUS "Found installed SuiteSparse: ${SUITESPARSE_LIBRARIES}")
    
    # Create interface targets for the installed SuiteSparse components
    if(NOT TARGET SuiteSparse::CHOLMOD)
        add_library(SuiteSparse::CHOLMOD INTERFACE IMPORTED)
        set_target_properties(SuiteSparse::CHOLMOD PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${CHOLMOD_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES "${CHOLMOD_LIBRARY};${CCOLAMD_LIBRARY};${CAMD_LIBRARY};${COLAMD_LIBRARY};${AMD_LIBRARY};${SUITESPARSE_CONFIG_LIBRARY};${LAPACK_LIBRARIES};${BLAS_LIBRARIES}"
        )
    endif()
    
    if(NOT TARGET SuiteSparse::AMD)
        add_library(SuiteSparse::AMD INTERFACE IMPORTED)
        set_target_properties(SuiteSparse::AMD PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${AMD_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES "${AMD_LIBRARY}"
        )
    endif()
    
    # Create a convenience target that includes all SuiteSparse components
    if(NOT TARGET SuiteSparse::SuiteSparse)
        add_library(SuiteSparse::SuiteSparse INTERFACE IMPORTED)
        set_target_properties(SuiteSparse::SuiteSparse PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${SUITESPARSE_INCLUDE_DIRS}"
            INTERFACE_LINK_LIBRARIES "${SUITESPARSE_LIBRARIES}"
        )
    endif()
else()
    message(STATUS "SuiteSparse not found, downloading and building from source...")
    
    # Find BLAS and LAPACK (required by SuiteSparse)
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    
    # Configure SuiteSparse build options before fetching
    # Build only minimal required packages for CHOLMOD
    set(SUITESPARSE_ENABLE_PROJECTS "suitesparse_config;amd;camd;colamd;ccolamd;cholmod" CACHE STRING "SuiteSparse projects to build")
    
    # Configure CHOLMOD modules
    set(CHOLMOD_GPL ON CACHE BOOL "Enable GPL modules (required for supernodal)")
    set(CHOLMOD_SUPERNODAL ON CACHE BOOL "Enable supernodal factorization")
    set(CHOLMOD_CHOLESKY ON CACHE BOOL "Enable Cholesky module")
    set(CHOLMOD_CAMD ON CACHE BOOL "Enable CAMD support")
    set(CHOLMOD_UTILITY ON CACHE BOOL "Enable Utility module")
    set(CHOLMOD_PARTITION OFF CACHE BOOL "Disable Partition module")
    set(CHOLMOD_MATRIXOPS OFF CACHE BOOL "Disable MatrixOps module")
    set(CHOLMOD_MODIFY OFF CACHE BOOL "Disable Modify module")
    
    # Disable CUDA/GPU support (CPU-only build)
    set(CHOLMOD_CUDA OFF CACHE BOOL "Disable CUDA GPU acceleration")
    
    # Disable demo/test builds
    set(SUITESPARSE_DEMOS OFF CACHE BOOL "Disable demos")
    
    # Disable Fortran if not needed
    set(SUITESPARSE_USE_FORTRAN OFF CACHE BOOL "Disable Fortran")
    
    # Enable building static libraries
    set(BUILD_STATIC_LIBS ON CACHE BOOL "Build static libraries")
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Don't build shared libraries")
    
    # Configure BLAS/LAPACK settings for SuiteSparse
    set(SUITESPARSE_USE_SYSTEM_BLAS ON CACHE BOOL "Use system BLAS")
    set(SUITESPARSE_USE_SYSTEM_LAPACK ON CACHE BOOL "Use system LAPACK")
    set(BLA_VENDOR "Generic" CACHE STRING "BLAS vendor")
    
    include(FetchContent)
    FetchContent_Declare(
        suitesparse
        GIT_REPOSITORY https://github.com/DrTimothyAldenDavis/SuiteSparse.git
        GIT_TAG v7.11.0
        GIT_SHALLOW TRUE
        EXCLUDE_FROM_ALL
    )
    
    message(STATUS "Fetching SuiteSparse...")
    FetchContent_MakeAvailable(suitesparse)
    message(STATUS "SuiteSparse configured with CHOLMOD and AMD")
    
    # Create alias targets for consistent naming
    # Note: We're building static libs, so targets are named *_static
    if(NOT TARGET SuiteSparse::CHOLMOD)
        if(TARGET CHOLMOD_static)
            add_library(SuiteSparse::CHOLMOD ALIAS CHOLMOD_static)
        elseif(TARGET CHOLMOD)
            add_library(SuiteSparse::CHOLMOD ALIAS CHOLMOD)
        endif()
    endif()
    
    if(NOT TARGET SuiteSparse::AMD)
        if(TARGET AMD_static)
            add_library(SuiteSparse::AMD ALIAS AMD_static)
        elseif(TARGET AMD)
            add_library(SuiteSparse::AMD ALIAS AMD)
        endif()
    endif()
    
    if(NOT TARGET SuiteSparse::COLAMD)
        if(TARGET COLAMD_static)
            add_library(SuiteSparse::COLAMD ALIAS COLAMD_static)
        elseif(TARGET COLAMD)
            add_library(SuiteSparse::COLAMD ALIAS COLAMD)
        endif()
    endif()
    
    if(NOT TARGET SuiteSparse::CAMD)
        if(TARGET CAMD_static)
            add_library(SuiteSparse::CAMD ALIAS CAMD_static)
        elseif(TARGET CAMD)
            add_library(SuiteSparse::CAMD ALIAS CAMD)
        endif()
    endif()
    
    if(NOT TARGET SuiteSparse::CCOLAMD)
        if(TARGET CCOLAMD_static)
            add_library(SuiteSparse::CCOLAMD ALIAS CCOLAMD_static)
        elseif(TARGET CCOLAMD)
            add_library(SuiteSparse::CCOLAMD ALIAS CCOLAMD)
        endif()
    endif()
    
    if(NOT TARGET SuiteSparse::SuiteSparseConfig)
        if(TARGET SuiteSparseConfig_static)
            add_library(SuiteSparse::SuiteSparseConfig ALIAS SuiteSparseConfig_static)
        elseif(TARGET SuiteSparseConfig)
            add_library(SuiteSparse::SuiteSparseConfig ALIAS SuiteSparseConfig)
        endif()
    endif()
    
    # Export include directories for easier use
    set(SUITESPARSE_INCLUDE_DIRS 
        ${suitesparse_SOURCE_DIR}/CHOLMOD/Include
        ${suitesparse_SOURCE_DIR}/AMD/Include
        ${suitesparse_SOURCE_DIR}/COLAMD/Include
        ${suitesparse_SOURCE_DIR}/CAMD/Include
        ${suitesparse_SOURCE_DIR}/CCOLAMD/Include
        ${suitesparse_SOURCE_DIR}/SuiteSparse_config
        CACHE PATH "SuiteSparse include directories"
    )
    
    message(STATUS "SuiteSparse will be built from source: ${suitesparse_SOURCE_DIR}")
endif()

