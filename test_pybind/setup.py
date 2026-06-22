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