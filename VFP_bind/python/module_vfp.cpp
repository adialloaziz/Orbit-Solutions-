#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <algorithm>
#include "../include/vfp.h"
#include "../include/utility.h"
namespace py = pybind11;

py::array_t<double> to_numpy(const std::vector<double>& values)
{
    py::array_t<double> result(values.size());
    std::copy(values.begin(), values.end(), result.mutable_data());
    return result;
}

PYBIND11_MODULE(module_vfp, m) 
{
    m.doc() = "VFP module";

    py::class_<VfpResults>(m, "VfpResults")
        .def_property_readonly("y", [](const VfpResults& result) { return to_numpy(result.y); })
        .def_property_readonly("t", [](const VfpResults& result) { return to_numpy(result.t); })
        .def_property_readonly("density", [](const VfpResults& result) { return to_numpy(result.density); })
        .def_property_readonly("mass", [](const VfpResults& result) { return to_numpy(result.mass); });
    
    m.def("init_vfp",
          [](int nq, int np, double Lq, double Lp, double rms_gauss_q, double rms_gauss_p) {
              std::vector<double> values = init_vfp(nq, np, Lq, Lp, rms_gauss_q, rms_gauss_p);
              return to_numpy(values);
          },
          "Initialize VFP distribution",
          py::arg("nq"), py::arg("np"), py::arg("Lq"), py::arg("Lp"),
          py::arg("rms_gauss_q"), py::arg("rms_gauss_p"));

    m.def(
        "integ_vfp",
        [](py::array_t<double, py::array::c_style> U0, double t1, double t2, int nt, int ntint,
           int nq, int np, double Lq, double Lp, double grand_L) {
            auto info = U0.request();
            if (info.ndim != 1 || info.size != static_cast<py::ssize_t>(nq) * np) {
                throw py::value_error("U0 must contain nq * np values");
            }
            return integ_vfp(U0.mutable_data(), t1, t2, nt, ntint, nq, np, Lq, Lp, grand_L);
        },
        "Integrate VFP equations over time interval [t1, t2]",
        py::arg("U0"), py::arg("t1"), py::arg("t2"), py::arg("nt"), py::arg("ntint"),
        py::arg("nq"), py::arg("np"), py::arg("Lq"), py::arg("Lp"), py::arg("grand_L"));
    
}