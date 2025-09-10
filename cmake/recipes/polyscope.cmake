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

if(TARGET polyscope::polyscope)
    return()
endif()

message(STATUS "Third-party: creating target 'polyscope::polyscope'")

include(FetchContent)
FetchContent_Declare(
        polyscope
        GIT_REPOSITORY https://github.com/nmwsharp/polyscope.git
        GIT_TAG v2.5.0
        GIT_SHALLOW TRUE
)

# Configure polyscope options before making it available
set(POLYSCOPE_BACKEND_OPENGL3_GLFW ON CACHE BOOL "Use OpenGL3 + GLFW backend" FORCE)
set(POLYSCOPE_BUILD_TESTS OFF CACHE BOOL "Build polyscope tests" FORCE)
set(POLYSCOPE_BUILD_EXAMPLES OFF CACHE BOOL "Build polyscope examples" FORCE)

FetchContent_MakeAvailable(polyscope)

# Create alias target for consistency
if(TARGET polyscope AND NOT TARGET polyscope::polyscope)
    add_library(polyscope::polyscope ALIAS polyscope)
endif()
