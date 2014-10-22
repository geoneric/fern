#include <Python.h>
#include "fern/core/thread_client.h"
#include "fern/python_extension/algorithm/numpy/algorithm.h"


static PyMethodDef methods[] = {
    {"add", fern::python::add, METH_VARARGS,
        "Add two arguments and return the result"},
    {"sqrt", fern::python::sqrt, METH_VARARGS,
        "Calculate the square root of the argument and return the result"},
    {nullptr, nullptr, 0, nullptr}
};


static fern::ThreadClient client;


PyMODINIT_FUNC init_fern_algorithm_numpy()
{
    PyObject* module = Py_InitModule("_fern_algorithm_numpy", methods);

    if(module) {
        // ...
    }
}
