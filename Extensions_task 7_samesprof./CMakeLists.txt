cmake_minimum_required(VERSION 3.10)
project(extension)

set(CMAKE_CXX_STANDARD 11)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include OpenCV directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable
add_executable(extension extension.cpp)

# Link OpenCV libraries
target_link_libraries(extension ${OpenCV_LIBS})

# Link OpenGL and GLUT frameworks (macOS-specific)
target_link_libraries(extension 
    "-framework OpenGL"
    "-framework GLUT"
)