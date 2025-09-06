set(REQUIRED_CMAKE_VERSION "3.9")

################################################################################
# CLI11
include(cli11)


if (${PROJECT_NAME}_WITH_API_DEMO)
    # Testing Example Creation
    add_executable(${PROJECT_NAME}_quick_start api_demos/quick_start.cpp)
    target_link_libraries(${PROJECT_NAME}_quick_start PUBLIC parth)
    target_link_libraries(${PROJECT_NAME}_quick_start PUBLIC CLI11::CLI11)

    #include eigen
    include(eigen)
    target_link_libraries(${PROJECT_NAME}_quick_start PUBLIC Eigen3::Eigen)


    #include libigl
    include(libigl)
    target_link_libraries(${PROJECT_NAME}_quick_start PUBLIC igl::core)
endif ()
