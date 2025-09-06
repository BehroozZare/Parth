# Examples CMakeLists.txt for Parth library
# This file is included from the main CMakeLists.txt when PARTH_SOLVER_WITH_EXAMPLES=ON

# Ensure we're being called from the main build system
if(NOT PROJECT_NAME STREQUAL "PARTH_SOLVER")
    message(FATAL_ERROR "Examples should be built as part of the main PARTH_SOLVER project")
endif()

# Set C++17 standard (inherited from parent)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Function to create an example executable
function(create_example target_name source_file)
    add_executable(${target_name} ${source_file})
    
    # Link with the main Parth library (which includes all dependencies)
    target_link_libraries(${target_name} PRIVATE ${PROJECT_NAME}_lib)
    
    # Include directories are already set up in the main library
    # But we add them explicitly for clarity
    target_include_directories(${target_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/Parth/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/LinSysSolver
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/DemoUtils/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/utils/includes
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
    )
    
    # Set output directory
    set_target_properties(${target_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/examples
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )
    
    # Copy any required data files
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/data)
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/data $<TARGET_FILE_DIR:${target_name}>/data
            COMMENT "Copying data files for ${target_name}"
        )
    endif()
    
    message(STATUS "Created example target: ${target_name}")
endfunction()

# API Demos
message(STATUS "Setting up API demos...")

# Permutation computation demo
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/api_demos/permutation_computation/permutation.cpp)
    create_example(permutation_demo 
        api_demos/permutation_computation/permutation.cpp)
    
    # Add a convenience target with shorter name
    add_custom_target(permutation DEPENDS permutation_demo)
endif()

# Cholesky integration demos (conditional on available libraries)
if(${PROJECT_NAME}_WITH_MKL AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/api_demos/cholesky_integration/mkl.cpp)
    create_example(cholesky_mkl_demo 
        api_demos/cholesky_integration/mkl.cpp)
    target_compile_definitions(cholesky_mkl_demo PRIVATE PARTH_WITH_MKL)
    add_custom_target(cholesky_mkl DEPENDS cholesky_mkl_demo)
endif()

if(${PROJECT_NAME}_WITH_ACCELERATE AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/api_demos/cholesky_integration/accelerate.cpp)
    create_example(cholesky_accelerate_demo 
        api_demos/cholesky_integration/accelerate.cpp)
    target_compile_definitions(cholesky_accelerate_demo PRIVATE PARTH_WITH_ACCELERATE)
    add_custom_target(cholesky_accelerate DEPENDS cholesky_accelerate_demo)
endif()

if(${PROJECT_NAME}_WITH_CHOLMOD AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/api_demos/cholesky_integration/cholmod.cpp)
    create_example(cholesky_cholmod_demo 
        api_demos/cholesky_integration/cholmod.cpp)
    target_compile_definitions(cholesky_cholmod_demo PRIVATE PARTH_WITH_CHOLMOD)
    add_custom_target(cholesky_cholmod DEPENDS cholesky_cholmod_demo)
endif()

# Search for any additional .cpp files in api_demos and create examples
file(GLOB_RECURSE ADDITIONAL_DEMO_SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/api_demos/*.cpp"
)

foreach(demo_source ${ADDITIONAL_DEMO_SOURCES})
    # Skip files we've already handled
    get_filename_component(demo_name ${demo_source} NAME_WE)
    if(NOT demo_name MATCHES "^(permutation|mkl|accelerate|cholmod)$")
        # Create target name from file path
        file(RELATIVE_PATH rel_path ${CMAKE_CURRENT_SOURCE_DIR}/api_demos ${demo_source})
        get_filename_component(demo_dir ${rel_path} DIRECTORY)
        if(demo_dir)
            string(REPLACE "/" "_" target_prefix ${demo_dir})
            set(target_name "${target_prefix}_${demo_name}_demo")
        else()
            set(target_name "${demo_name}_demo")
        endif()
        
        create_example(${target_name} api_demos/${rel_path})
    endif()
endforeach()

# IPC Example (if it has source files)
file(GLOB IPC_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/ipc_example/*.cpp")
if(IPC_SOURCES)
    message(STATUS "Found IPC example sources")
    foreach(ipc_source ${IPC_SOURCES})
        get_filename_component(ipc_name ${ipc_source} NAME_WE)
        create_example(ipc_${ipc_name}_demo ${ipc_source})
    endforeach()
endif()

# Patch Remeshing Example (if it has source files)
file(GLOB PATCH_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/patch_remeshing/*.cpp")
if(PATCH_SOURCES)
    message(STATUS "Found patch remeshing example sources")
    foreach(patch_source ${PATCH_SOURCES})
        get_filename_component(patch_name ${patch_source} NAME_WE)
        create_example(patch_${patch_name}_demo ${patch_source})
    endforeach()
endif()

# Create a custom target that builds all examples
add_custom_target(all_examples)

# Get all example targets and add them to all_examples
get_property(all_targets DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY BUILDSYSTEM_TARGETS)
foreach(target ${all_targets})
    if(target MATCHES ".*_demo$")
        add_dependencies(all_examples ${target})
    endif()
endforeach()

# Add helpful aliases
add_custom_target(examples DEPENDS all_examples)

# Print summary
message(STATUS "Examples configuration complete:")
message(STATUS "  - Permutation computation demo: ENABLED")
if(${PROJECT_NAME}_WITH_MKL)
    message(STATUS "  - MKL Cholesky demo: ENABLED")
endif()
if(${PROJECT_NAME}_WITH_ACCELERATE)
    message(STATUS "  - Accelerate Cholesky demo: ENABLED") 
endif()
if(${PROJECT_NAME}_WITH_CHOLMOD)
    message(STATUS "  - CHOLMOD Cholesky demo: ENABLED")
endif()
message(STATUS "  - Output directory: ${CMAKE_BINARY_DIR}/bin/examples")
message(STATUS "  - Build all examples: 'make all_examples' or 'make examples'")
message(STATUS "  - Build individual examples: 'make permutation', 'make cholesky_mkl', etc.")