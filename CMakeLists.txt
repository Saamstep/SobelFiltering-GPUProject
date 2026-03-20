cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

if(NOT DEFINED SOBEL_CUDA_ARCHITECTURES)
    # This workspace is currently running on an RTX 5080 (SM 12.0). Defaulting to
    # native SASS for that GPU avoids PTX JIT failures when the installed driver
    # lags the CUDA toolkit version.
    set(SOBEL_CUDA_ARCHITECTURES "120-real" CACHE STRING "CUDA architectures for sobel_gpu")
endif()

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
# CUDA target
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

    if(MSVC)
        target_compile_options(sobel_gpu PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/O2 /Oi /Zc:preprocessor>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
        )
    endif()

    if(CUDA_cudart_LIBRARY)
        get_filename_component(_cuda_lib_dir "${CUDA_cudart_LIBRARY}" DIRECTORY)
        get_filename_component(_cuda_lib_parent_dir "${_cuda_lib_dir}" DIRECTORY)
        get_filename_component(_cuda_root_dir "${_cuda_lib_parent_dir}" DIRECTORY)
        set_property(TARGET sobel_gpu PROPERTY VS_GLOBAL_CudaToolkitDir "${_cuda_root_dir}")
    endif()

    set_target_properties(sobel_gpu PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${SOBEL_CUDA_ARCHITECTURES}"
    )

    copy_opencv_runtime_dlls(sobel_gpu)
endif()
