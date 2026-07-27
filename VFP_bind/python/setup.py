from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "module_vfp",
        ["./module_vfp.cpp",
         "../src/VFP.cpp",
         "../src/utility.cpp"],

         extra_compile_args=[
             "-std=c++17",
             "-O3",
             ],
        libraries=["gsl", "gslcblas", "fftw3", "m"],

    ),
]
setup(
    name="module_vfp",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)