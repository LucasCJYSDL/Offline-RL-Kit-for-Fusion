from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import os

current_working_directory = os.getcwd()
# Define extensions
extensions = [
    Extension(
        "search_tree",  # Name of the module
        [current_working_directory + "/search_tree.pyx", current_working_directory + "/lib/cnode.cpp",
         current_working_directory + "/lib/cminimax.cpp"],  # Source files
        include_dirs=[current_working_directory + "/lib/"],  # Include directories for header files
        language="c++",  # Specify the language
        extra_compile_args=["-std=c++11"],  # Optional: compile args, like C++ version
    )
]

# Setup function
setup(
    name="search_tree",
    ext_modules=cythonize(extensions)
)

# please use "python setup.py build_ext --inplace" to compile the cython and c source codes to a python extension