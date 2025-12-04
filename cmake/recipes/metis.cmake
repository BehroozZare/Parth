# METIS recipe: find installed or download and build
#
# This recipe first tries to find an installed METIS library.
# If not found, it downloads and builds METIS from source.

if(TARGET metis)
    return()
endif()

# First, try to find an installed METIS
find_package(METIS QUIET)

if(METIS_INCLUDES AND METIS_LIBRARIES)
    message(STATUS "Found installed METIS: ${METIS_LIBRARIES}")
    
    # Create an interface target for the installed METIS
    add_library(metis INTERFACE)
    target_include_directories(metis INTERFACE ${METIS_INCLUDES})
    target_link_libraries(metis INTERFACE ${METIS_LIBRARIES})
else()
    message(STATUS "METIS not found, downloading and building from source...")
    
    # Set IDXTYPEWIDTH and REALTYPEWIDTH before fetching
    set(IDXTYPEWIDTH 32 CACHE STRING "Width of integer type for METIS")
    set(REALTYPEWIDTH 32 CACHE STRING "Width of real type for METIS")

    include(FetchContent)
    FetchContent_Declare(
        metis
        GIT_REPOSITORY https://github.com/scivision/METIS.git
        GIT_TAG d4a3aac2a3a0efc18e1de24ae97302ed510f43c7
    )
    FetchContent_MakeAvailable(metis)

    # Include directories and definitions
    target_include_directories(metis INTERFACE
        $<BUILD_INTERFACE:${metis_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    target_compile_definitions(metis INTERFACE
        IDXTYPEWIDTH=${IDXTYPEWIDTH}
        REALTYPEWIDTH=${REALTYPEWIDTH}
    )

    message(STATUS "METIS will be built from source: ${metis_SOURCE_DIR}")
endif()

