#define PY_SSIZE_T_CLEAN
#include <iostream>
#include </home/lamlam/ENTER/include/python3.10/Python.h>
#include </home/lamlam/ENTER/lib/python3.10/site-packages/pybind11/include/pybind11/embed.h>

using namespace std;
namespace py = pybind11;

int main() {
    py::scoped_interpreter guard{};

    // Execute Python code to define a function
    py::exec(R"(
        print('Hello, Python!')
    )");

    // Execute Python code to define a function
    py::exec(R"(
        def add_numbers(a, b):
            return a + b
    )");

    // Call the Python function from C++
    py::object add_numbers = py::module::import("__main__").attr("add_numbers");
    int result = add_numbers(2, 3).cast<int>();
    std::cout << "Result: " << result << std::endl;

    /*
    //Initialize Python interpreter
    Py_Initialize();

    //Run lines of Python code directly
    PyRun_SimpleString("print('Begin')");
    PyRun_SimpleString("from time import time,ctime\n"
                       "print('Today is', ctime(time()))\n");
    
    //Run a python file
    FILE* file = fopen("binding.py", "r");
    PyRun_SimpleFile(file, "binding.py");
    fclose(file);
    
    PyObject* pName, * pModule, * pFunc, * pArgs = nullptr, * pValue; 
    //Specify file name to be imported 
    pName = PyUnicode_FromString("binding");
    //Import the file
    pModule = PyImport_Import(pName);
    //Find the function in the python module file
    pFunc = PyObject_GetAttrString(pModule, "hello");
    //Call the function
    PyObject_CallObject(pFunc, NULL);//---Call hello function

    Py_Finalize();
    return(0);
    
    /*
    
    pName = PyUnicode_FromString((char*)"Sample"); 
    pModule = PyImport_Import(pName); 
    pFunc = PyObject_GetAttrString(pModule, (char*)"fun"); 
    pValue = PyObject_CallObject(pFunc, pArgs);
    */
}