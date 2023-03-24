cmake_minimum_required(VERSION 3.16)
project(QP)
include(FindBLAS)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")




add_executable(QP main.cpp algorithm.cpp algorithm.h utils.h)

#find_package(BLAS REQUIRED)

#if (BLAS_FOUND)
#    target_compile_options(conv_clustering PRIVATE ${BLAS_LINKER_FLAGS})
#    target_link_libraries(conv_clustering ${BLAS_LIBRARIES})
#endif ()

include_directories(/usr/local/include/eigen3)

#FIND_PACKAGE(MKL REQUIRED)
#if (MKL_FOUND)
#    include_directories(/opt/intel/oneapi/mkl/2022.1.0/include)
#    link_directories(/opt/intel/oneapi/mkl/2022.1.0/lib/intel64)
#endif ()

#target_link_libraries(conv_clustering  -Wl,--start-group /opt/intel/oneapi/mkl/2022.1.0/lib/intel64/libmkl_intel_ilp64.a /opt/intel/oneapi/mkl/2022.1.0/lib/intel64/libmkl_gnu_thread.a /opt/intel/oneapi/mkl/2022.1.0/lib/intel64/libmkl_core.a -Wl,--end-group -lgomp -lpthread -lm -ldl)

FIND_PACKAGE(OpenMP REQUIRED)
if (OpenMP_FOUND)
    set(CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS}")
endif ()