
cmake_minimum_required(VERSION 2.8)
find_package(CUDA)
find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
include_directories(
    ${GLUT_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIR}
    )
message(${OPENGL_INCLUDE_DIR})
set(CMAKE_CXX_FLAGS "-std=c++11")

file( GLOB  hdr *.hpp *.h )
file( GLOB  cu  *.cu)
SET (to_bw_files main.cpp InitShader.cpp )

#add_executable(practica InitShader.cpp)

CUDA_ADD_EXECUTABLE(practica ${to_bw_files} ${hdr} ${cu})

target_link_libraries(practica
${OPENGL_LIBRARY}
${GLUT_LIBRARY}
${GLEW_LIBRARY}
${OpenCV_LIBS})
