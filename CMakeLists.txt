cmake_minimum_required(VERSION 3.10)
project(rnnt_release LANGUAGES CXX)

# ==== CUDA DETECTION ====
set(CUDA_FOUND FALSE)
set(CUDA_TOOLKIT_VERSION "0.0")

find_package(CUDA QUIET)

if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
    find_package(CUDAToolkit QUIET)
    
    if(CUDAToolkit_FOUND OR CUDA_FOUND)
        set(CUDA_FOUND TRUE)
        if(CUDAToolkit_FOUND)
            set(CUDA_TOOLKIT_VERSION "${CUDAToolkit_VERSION}")
        elseif(DEFINED CUDA_VERSION AND NOT "${CUDA_VERSION}" STREQUAL "")
            set(CUDA_TOOLKIT_VERSION "${CUDA_VERSION}")
        endif()
    endif()
else()
    if(CUDA_FOUND AND DEFINED CUDA_VERSION AND NOT "${CUDA_VERSION}" STREQUAL "")
        set(CUDA_TOOLKIT_VERSION "${CUDA_VERSION}")
    endif()
endif()

# ==== USER OPTIONS ====
option(WITH_GPU "Build with GPU (CUDA) support" ${CUDA_FOUND})
option(WITH_OMP "Build with OpenMP support" ON)
option(USE_NAIVE_KERNEL "Use naive alpha-beta kernel" OFF)
option(DEBUG_TIME "Output kernel time" OFF)
option(DEBUG_KERNEL "Output alpha beta debug" OFF)

# ==== GPU BUILD CONFIGURATION ====
if(WITH_GPU)
    if(NOT CUDA_FOUND)
        message(WARNING "WITH_GPU=ON was requested, but no compatible CUDA toolkit was found. Falling back to CPU-only build.")
        set(WITH_GPU OFF CACHE BOOL "Build with GPU (CUDA) support" FORCE)
        set(USE_CUDA FALSE)
    else()
        set(USE_CUDA TRUE)
        message(STATUS "Building with CUDA ${CUDA_TOOLKIT_VERSION} support")
    endif()
else()
    set(USE_CUDA FALSE)
    if(CUDA_FOUND)
        message(STATUS "CUDA ${CUDA_TOOLKIT_VERSION} detected but GPU support disabled (WITH_GPU=OFF)")
    else()
        message(STATUS "Building CPU-only version (no CUDA toolkit found)")
    endif()
endif()

# ==== OPENMP CONFIGURATION ====
if(WITH_OMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP support enabled")
    else()
        message(WARNING "OpenMP requested but not found")
        set(WITH_OMP OFF)
    endif()
endif()

# ==== COMPILER SETTINGS ====
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
endif()

if(WITH_OMP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

if(APPLE)
    add_definitions(-DAPPLE)
    execute_process(COMMAND uname -v OUTPUT_VARIABLE DARWIN_VERSION)
    string(REGEX MATCH "[0-9]+" DARWIN_VERSION "${DARWIN_VERSION}")
    message(STATUS "Darwin version: ${DARWIN_VERSION}")
    if(${DARWIN_VERSION} LESS 15)
        set(CMAKE_SKIP_RPATH TRUE)
    endif()
else()
    set(CMAKE_SKIP_RPATH TRUE)
endif()

include_directories(include)

# ==== PREPROCESSOR DEFINITIONS ====
if(USE_NAIVE_KERNEL)
    add_definitions(-DUSE_NAIVE_KERNEL)
endif()

if(DEBUG_TIME)
    add_definitions(-DDEBUG_TIME)
endif()

if(DEBUG_KERNEL)
    add_definitions(-DDEBUG_KERNEL)
endif()

if(NOT WITH_OMP)
    add_definitions(-DRNNT_DISABLE_OMP)
endif()

# ==== CUDA CONFIGURATION ====
if(USE_CUDA)
    if(WITH_OMP)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fopenmp")
    endif()

    if(NOT (CUDA_VERSION GREATER 10.2))
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30 -O2")
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_35,code=sm_35")
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50")
    endif()

    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_52,code=sm_52")
    
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES")
    endif()

    if(CUDA_VERSION GREATER 7.6)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_60,code=sm_60")
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61")
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_62,code=sm_62")
    endif()

    if(CUDA_VERSION GREATER 8.9)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_70,code=sm_70")
    endif()

    if(CUDA_VERSION GREATER 9.9)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75")
    endif()

    if(CUDA_VERSION GREATER 11.0)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_80,code=sm_80")
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_86,code=sm_86")
    endif()

    if(NOT APPLE)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --std=c++17")
    endif()

    message(STATUS "CUDA NVCC flags: ${CUDA_NVCC_FLAGS}")
endif()

# ==== SOURCE FILE PREPARATION ====
set(RNNT_ENTRYPOINT_SRC src/rnnt_entrypoint.cu)
if(NOT USE_CUDA)
    configure_file(
        ${RNNT_ENTRYPOINT_SRC}
        ${CMAKE_CURRENT_BINARY_DIR}/rnnt_entrypoint_cpu.cpp
        COPYONLY
    )
    set(RNNT_ENTRYPOINT_SRC ${CMAKE_CURRENT_BINARY_DIR}/rnnt_entrypoint_cpu.cpp)
endif()

# ==== MAIN LIBRARY TARGET ====
if(USE_CUDA)
    message(STATUS "Building shared library with GPU support")
    
    cuda_add_library(warprnnt SHARED ${RNNT_ENTRYPOINT_SRC})
    
    if(NOT Torch_FOUND)
        target_link_libraries(warprnnt ${CUDA_curand_LIBRARY})
    endif()
    
    if(WITH_OMP AND OpenMP_CXX_FOUND)
        target_link_libraries(warprnnt ${OpenMP_CXX_LIBRARIES})
    endif()
else()
    message(STATUS "Building shared library with CPU-only support")
    add_library(warprnnt SHARED ${RNNT_ENTRYPOINT_SRC})
    
    if(WITH_OMP AND OpenMP_CXX_FOUND)
        target_link_libraries(warprnnt PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

set_target_properties(warprnnt PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

if(WIN32)
    set_target_properties(warprnnt PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# Use include_directories (global) since we're mixing legacy/modern approaches
# target_include_directories(warprnnt PRIVATE include) # Don't use this with legacy CUDA

# ==== TEST EXECUTABLES ====
add_executable(test_cpu tests/test_cpu.cpp tests/random.cpp)
target_link_libraries(test_cpu warprnnt)
set_target_properties(test_cpu PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

add_executable(test_time tests/test_time.cpp tests/random.cpp)
target_link_libraries(test_time warprnnt)
set_target_properties(test_time PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

if(USE_CUDA)
    cuda_add_executable(test_time_gpu tests/test_time.cu tests/random.cpp)
    target_link_libraries(test_time_gpu warprnnt ${CUDA_curand_LIBRARY})
    set_target_properties(test_time_gpu PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

    cuda_add_executable(test_gpu tests/test_gpu.cu tests/random.cpp)
    target_link_libraries(test_gpu warprnnt ${CUDA_curand_LIBRARY})
    set_target_properties(test_gpu PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")
endif()

# ==== INSTALL TARGETS ====
install(TARGETS warprnnt
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES include/rnnt.h DESTINATION include)

# ==== BUILD SUMMARY ====
message(STATUS "")
message(STATUS "=== Build Configuration Summary ===")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA Available: ${CUDA_FOUND}")
if(CUDA_FOUND)
    message(STATUS "CUDA Version: ${CUDA_TOOLKIT_VERSION}")
endif()
message(STATUS "GPU Support: ${USE_CUDA}")
if(USE_CUDA)
    message(STATUS "CUDA NVCC Flags: ${CUDA_NVCC_FLAGS}")
endif()
message(STATUS "OpenMP Support: ${WITH_OMP}")
message(STATUS "Naive Kernel: ${USE_NAIVE_KERNEL}")
message(STATUS "Debug Time: ${DEBUG_TIME}")
message(STATUS "Debug Kernel: ${DEBUG_KERNEL}")
message(STATUS "=====================================")
message(STATUS "")
