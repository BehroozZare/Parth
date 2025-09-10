# Polyscope Integration for Parth Project

This document describes the Polyscope integration added to the Parth project for 3D visualization capabilities.

## What was added

### 1. Polyscope Recipe (`cmake/recipes/polyscope.cmake`)
- Follows the same pattern as other dependencies (eigen, libigl, etc.)
- Uses FetchContent to download Polyscope from GitHub
- Configures Polyscope with OpenGL3 + GLFW backend
- Disables tests and examples to reduce build time
- Creates `polyscope::polyscope` target for linking

### 2. Utils Library (`dev/utils/`)
A new utility library was created with the following components:

#### `vertex_colored_mesh.h/cpp`
- `VertexColoredMesh` class for handling mesh data with colors and scalars
- Methods to set vertices, faces, colors, and scalar values
- Basic visualization interface (placeholder for polyscope integration)

#### `polyscope_utils.h/cpp`
- High-level wrapper functions for common Polyscope operations
- `initialize()` - Initialize Polyscope
- `show()` - Display the Polyscope GUI
- `registerSurfaceMesh()` - Register a mesh with Polyscope
- `addVertexColors()` - Add vertex colors to a mesh
- `addVertexScalars()` - Add vertex scalars to a mesh

### 3. Updated CMakeLists.txt
- Added `include(polyscope)` to load the polyscope recipe
- Created `parth_dev_utils` library target
- Linked all executables with polyscope and the utils library
- Added example executable `PARTH_polyscope_example`

### 4. Example Usage (`polyscope_example.cpp`)
Demonstrates how to:
- Initialize Polyscope
- Create simple mesh data
- Register meshes with Polyscope
- Add vertex colors and scalars
- Use the utility classes

## Building

After these changes, building the project will automatically:
1. Download and build Polyscope as a dependency
2. Build the utils library with Polyscope integration
3. Link all executables with Polyscope

Build commands:
```bash
mkdir build && cd build
cmake ..
make
```

## Usage Example

```cpp
#include "utils/polyscope_utils.h"
#include <Eigen/Core>

int main() {
    // Initialize polyscope
    PARTH::utils::polyscope_utils::initialize();
    
    // Create mesh data
    Eigen::MatrixXd vertices(3, 3);
    vertices << 0.0, 0.0, 0.0,
                1.0, 0.0, 0.0,
                0.5, 1.0, 0.0;
    
    Eigen::MatrixXi faces(1, 3);
    faces << 0, 1, 2;
    
    // Register with polyscope
    PARTH::utils::polyscope_utils::registerSurfaceMesh("triangle", vertices, faces);
    
    // Show GUI
    PARTH::utils::polyscope_utils::show();
    
    return 0;
}
```

## Available Executables

After building, you'll have these new executables:
- `PARTH_polyscope_example` - Demonstrates polyscope integration
- Existing executables now have polyscope available for visualization

## Integration with Existing Code

You can now add visualization to any of your existing analysis tools by:

1. Including the polyscope utils headers:
   ```cpp
   #include "utils/polyscope_utils.h"
   ```

2. Initializing polyscope:
   ```cpp
   PARTH::utils::polyscope_utils::initialize();
   ```

3. Registering your mesh data:
   ```cpp
   PARTH::utils::polyscope_utils::registerSurfaceMesh("mesh_name", vertices, faces);
   ```

4. Adding visualization data (colors, scalars, etc.):
   ```cpp
   PARTH::utils::polyscope_utils::addVertexColors("mesh_name", "colors", color_matrix);
   ```

5. Showing the GUI:
   ```cpp
   PARTH::utils::polyscope_utils::show();
   ```

## Notes

- Polyscope is a header-only visualization library for 3D data
- It provides interactive GUI for mesh inspection and data visualization
- The integration follows the same pattern as other dependencies in the project
- All linking is handled automatically through CMake
