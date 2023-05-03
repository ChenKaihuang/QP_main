# example.py

from cffi import FFI

# Create a FFI object
ffi = FFI()

# Define the C interface for your C library
ffi.cdef("""
    // Declare the necessary types and functions from your C library
    // ...

    // Declare the necessary types and functions from the third-party library
""")

# Set the source and libraries during compilation
# ffi.set_source("_QP", r"""
#     #include "algorithm.h"
# """, libraries=['gfortran'])  # Include the 'm' library, which is the math library

# Load your C library and the third-party library
# third_party_lib = ffi.dlopen("./libpardiso800-GNU831-X86-64-March18-2023.so")  # Assumes third_party_lib.so is in the same directory
my_lib = ffi.dlopen("./QP_library_save.so")  # Assumes my_lib.so is in the same directory

# Use functions and data types from your C library and the third-party library
# ...
