cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(SobelProject VERSION 1.0 LANGUAGES CXX CUDA)

# -----------------------------
# C++ standard
# -----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures")
endif()

# -----------------------------
# Find OpenCV
# -----------------------------
if(WIN32)
    set(OpenCV_DIR "C:/opencv/build/x64/vc16/lib")
endif()

find_package(OpenCV)

message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

function(copy_opencv_runtime_dlls target_name)
    if(WIN32 AND OpenCV_INCLUDE_DIRS)
        list(GET OpenCV_INCLUDE_DIRS 0 _opencv_first_include_dir)
        get_filename_component(_opencv_root_dir "${_opencv_first_include_dir}" DIRECTORY)
        set(_opencv_bin_dir "${_opencv_root_dir}/x64/vc16/bin")

        file(GLOB _opencv_runtime_dlls "${_opencv_bin_dir}/*.dll")
        if(_opencv_runtime_dlls)
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${_opencv_runtime_dlls}
                    $<TARGET_FILE_DIR:${target_name}>
                COMMAND_EXPAND_LISTS
            )
        else()
            message(WARNING "No OpenCV runtime DLLs found in ${_opencv_bin_dir}")
        endif()
    endif()
endfunction()

# -----------------------------
# CPU-only target
# -----------------------------
add_executable(sobel_cpu
    main.cpp
    # CPU Sobel does not call CUDA
)

target_include_directories(sobel_cpu
    PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(sobel_cpu PRIVATE ${OpenCV_LIBS})

if(MSVC)
    target_compile_options(sobel_cpu PRIVATE /O2 /Oi)
else()
    target_compile_options(sobel_cpu PRIVATE -O3 -march=native)
endif()

copy_opencv_runtime_dlls(sobel_cpu)

# -----------------------------
# CUDA target (optional)
# -----------------------------
if(WIN32)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    add_executable(sobel_gpu
        main.cpp
        sobel.cu  # your CUDA kernels
    )

    target_include_directories(sobel_gpu
        PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_link_libraries(sobel_gpu PRIVATE ${OpenCV_LIBS} CUDA::cudart)
    target_compile_definitions(sobel_gpu PRIVATE SOBEL_ENABLE_CUDA=1)

    if(CUDA_cudart_LIBRARY)
        get_filename_component(_cuda_lib_dir "${CUDA_cudart_LIBRARY}" DIRECTORY)
        get_filename_component(_cuda_lib_parent_dir "${_cuda_lib_dir}" DIRECTORY)
        get_filename_component(_cuda_root_dir "${_cuda_lib_parent_dir}" DIRECTORY)
        set_property(TARGET sobel_gpu PROPERTY VS_GLOBAL_CudaToolkitDir "${_cuda_root_dir}")
    endif()

    set_target_properties(sobel_gpu PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )

    copy_opencv_runtime_dlls(sobel_gpu)
endif()
