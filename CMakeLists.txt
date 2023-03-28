cmake_minimum_required(VERSION 3.16)
project(QP)
include(FindBLAS)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

add_executable(QP main.cpp algorithm.cpp algorithm.h utils.h)

include_directories(/usr/local/include/eigen3)

# Add PARDISO library
find_library(PARDISO_LIB libpardiso800-GNU831-X86-64-March18-2023.so PATHS ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(QP PRIVATE ${PARDISO_LIB})

# Add LAPACK/BLAS libraries
find_library(LAPACK_LIB liblapack.so)
find_library(BLAS_LIB libblas.so)
target_link_libraries(QP PRIVATE ${LAPACK_LIB} ${BLAS_LIB})

# Add additional libraries and flags
target_link_libraries(QP PRIVATE gfortran pthread m gomp)