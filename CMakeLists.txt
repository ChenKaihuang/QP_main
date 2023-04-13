cmake_minimum_required(VERSION 3.16)
project(QP)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})

# Set the name and type of your shared library
add_library(QP_library SHARED
        algorithm.cpp
        # ... add more source files as needed
        QP_wrapper.cpp)

# Link against Python libraries
target_link_libraries(QP_library PRIVATE Python3::Python)

# Set the output name and extension of the Python module
set_target_properties(QP_library PROPERTIES
        OUTPUT_NAME QP_library
        SUFFIX .so  # Use .dll for Windows or .pyd for Python on Windows
        )

# Set the target directory for the Python module
set_target_properties(QP_library PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )

# Set the compilation options
target_compile_options(QP_library PRIVATE -Wall -Xlinker -export-dynamic)

# Install the Python module to the site-packages directory
install(TARGETS QP_library
        LIBRARY DESTINATION ${Python3_SITELIB})



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
target_link_libraries(QP PRIVATE gfortran pthread m)
FIND_PACKAGE(OpenMP REQUIRED)
if (OpenMP_FOUND)
    set(CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS}")
endif ()