#
# Copyright 2020 Adobe. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.
#
if(TARGET Eigen3::Eigen)
    return()
endif()

# option(EIGEN_WITH_MKL "Use Eigen with MKL" OFF)

if(EIGEN_ROOT)
    message(STATUS "Third-party: creating target 'Eigen3::Eigen' for external path: ${EIGEN_ROOT}")
    add_library(Eigen3_Eigen INTERFACE)
    add_library(Eigen3::Eigen ALIAS Eigen3_Eigen)
    
    include(GNUInstallDirs)
    target_include_directories(Eigen3_Eigen SYSTEM INTERFACE
            $<BUILD_INTERFACE:${EIGEN_ROOT}>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
else()
    message(STATUS "Third-party: fetching 'Eigen3::Eigen'")

    include(FetchContent)
    FetchContent_Declare(
            eigen
            GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
            GIT_TAG 3.4.0
            GIT_SHALLOW TRUE
    )
    
    # Disable building tests/docs
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    
    FetchContent_MakeAvailable(eigen)
    # Eigen's CMakeLists.txt creates the Eigen3::Eigen target
endif()
