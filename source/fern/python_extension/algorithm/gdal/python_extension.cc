#include <Python.h>
#include "fern/core/thread_client.h"
#include "fern/python_extension/algorithm/gdal/algorithm.h"


static PyMethodDef methods[] = {
    {"add", fern::python::add, METH_VARARGS,
        "Add two arguments and return the result"},
    {"slope", fern::python::slope, METH_VARARGS,
        "Calculate the slope and return the result"},
    {nullptr, nullptr, 0, nullptr}
};


static fern::ThreadClient thread_client;


PyMODINIT_FUNC init_fern_algorithm_gdal(
    void)
{
    PyObject* module = Py_InitModule("_fern_algorithm_gdal", methods);

    if(module) {
        // ...
    }
}
