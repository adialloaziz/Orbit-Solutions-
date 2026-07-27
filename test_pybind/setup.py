# This is a setup.py file for building a Python extension module using pybind11. It defines the extension module and specifies the source files needed to compile it. 
# The setup function is called to configure the build process, including the name of the package and the command class for building extensions.
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "module",  #Name of the module-the cpp-code-
        ["./src/module.cpp"], #Path to the cpp code

    ),
]

setup(
    name="test_pybind",
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    
)