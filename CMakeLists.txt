cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(SobelProject VERSION 1.0 LANGUAGES CXX CUDA)

# -----------------------------
# C++ standard
# -----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------
# Find OpenCV
# -----------------------------
if(WIN32)
    set(OpenCV_DIR "C:/opencv/build/x64/vc16/lib")
endif()

find_package(OpenCV)

message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

# -----------------------------
# Sources
# -----------------------------
set(SOURCES
    main.cpp
    # implementation.cpp
    # sobel.cpp   # optional if you separate CUDA stuff
)

# -----------------------------
# CUDA support
# -----------------------------
if(WIN32)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    list(APPEND SOURCES sobel.cu)
endif()

# -----------------------------
# Executable
# -----------------------------
add_executable(sobel_app ${SOURCES})

if(WIN32)
    enable_language(CUDA)
    target_sources(sobel_app PRIVATE sobel.cu)
    set_target_properties(sobel_app PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
endif()

# -----------------------------
# Include dirs
# -----------------------------
target_include_directories(sobel_app
    PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# -----------------------------
# Link libraries
# -----------------------------
target_link_libraries(sobel_app
    PRIVATE
    ${OpenCV_LIBS}
)

if(WIN32)
    target_link_libraries(sobel_app PRIVATE CUDA::cudart)
endif()

# -----------------------------
# Compiler optimization
# -----------------------------
if(MSVC)
    target_compile_options(sobel_app PRIVATE /O2 /Oi)
else()
    target_compile_options(sobel_app PRIVATE -O3 -march=native)
endif()