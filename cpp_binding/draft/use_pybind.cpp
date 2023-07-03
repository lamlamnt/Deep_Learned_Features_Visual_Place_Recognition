#include </home/lamlam/ENTER/lib/python3.10/site-packages/pybind11/include/pybind11/embed.h>
#include <iostream>

namespace py = pybind11;

void callPythonFile(const std::string& filename, const std::vector<int>& arguments) {
    py::scoped_interpreter guard{};

    py::module main = py::module::import("__main__");
    py::object global = main.attr("__dict__");

    py::object result = py::eval_file(filename, global);

    // Call a function in the Python file with the arguments
    py::object myFunction = global["my_function"];
    py::object pyArgs = py::cast(arguments);
    py::object pyResult = myFunction(pyArgs);
}

PYBIND11_MODULE(example, m) {
    m.def("call_python_file", &callPythonFile, "Call a Python file with arguments");
}

int main() {
    callPythonFile("binding.py",[1,2]);
    return 0;
}