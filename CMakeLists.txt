cmake_minimum_required(VERSION 3.18)
project(SobelProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(sobel_app main.cpp)
target_include_directories(sobel_app PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(sobel_app PRIVATE ${OpenCV_LIBS})

if(NOT MSVC)
    target_compile_options(sobel_app PRIVATE -O3 -march=native)
endif()

if(WIN32)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    list(APPEND SOURCES sobel.cu)
    target_link_libraries(sobel_app PRIVATE CUDA::cudart)
endif()