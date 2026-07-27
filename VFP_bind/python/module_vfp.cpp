#include <pybind11/pybind11.h>
#include "../include/vfp.h"
namespace py = pybind11;

PYBIND11_MODULE(module_vfp, m) {
    m.doc() = "VFP module";

    m.def(
        "integ_vfp", &integ_vfp,
         "Integrate VFP equations over time interval [t1, t2]",
         py::arg("t1"), py::arg("t2")
        );
}