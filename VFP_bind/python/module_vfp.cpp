#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include "../include/vfp.h"
#include "../include/utility.h"
namespace py = pybind11;

PYBIND11_MODULE(module_vfp, m) {
    m.doc() = "VFP module";

    py::class_<VfpResults>(m, "VfpResults")
        .def_readonly("y", &VfpResults::y)
        .def_readonly("t", &VfpResults::t)
        .def_readonly("density", &VfpResults::density)
        .def_readonly("mass", &VfpResults::mass);
    m.def(
        "integ_vfp", &integ_vfp,
         "Integrate VFP equations over time interval [t1, t2]",
         py::arg("t1"), py::arg("t2"), py::arg("nt"), py::arg("ntint")
        );
    
    m.def("init_vfp", &init_vfp, "Initialize VFP distribution",
          py::arg("U0"), py::arg("nq"), py::arg("np"), py::arg("Lq"), py::arg("Lp"),
          py::arg("rms_gauss_q"), py::arg("rms_gauss_p"));
}