#include <pybind11/pybind11.h>
namespace py = pybind11;

// A simple test function that adds two numbers
int add(int a, int b) {
    return a + b;
}

double area_rect(double length, double width){

    return length * width;
}
std:: string hello(std:: string name){
    return "Hello, " +  name + " !";
}

PYBIND11_MODULE(module, m){
    m.doc() = "Documentation of the module";
    m.def("add", &add, 
    "Add two integers",
    py::arg("a"), py::arg("b")
        );

    m.def("area_rect", &area_rect,
        "Compute the area of a rectangle",
        py::arg("length"), py::arg("width"));
    
    m.def("hello", &hello,
        "Say Hello to someone",
        py::arg("name"));
}