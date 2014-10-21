#include <Python.h>
#include <numpy/arrayobject.h>
#include "fern/core/thread_client.h"
#include "fern/python_extension/algorithm/numpy/algorithm.h"


namespace fern {

static void init_numpy()
{
    import_array();
}


#define TRY_BINARY_ALGORITHM(                                               \
    algorithm,                                                              \
    type1,                                                                  \
    type2)                                                                  \
if(!result && Py##type1##_Check(value1_object) && Py##type2##_Check(        \
        value2_object)) {                                                   \
    result = (PyObject*)algorithm(                                          \
        (Py##type1##Object const*)value1_object,                            \
        (Py##type2##Object const*)value2_object);                           \
}


static PyObject* add(
    PyObject* /* self */,
    PyObject* arguments)
{
    init_numpy();

    // Parse PyObject instances, verify their types. Switch on argument type.

    PyObject* value1_object;
    PyObject* value2_object;

    if(!PyArg_ParseTuple(arguments, "OO", &value1_object, &value2_object)) {
        return nullptr;
    }

    PyObject* result{nullptr};

    TRY_BINARY_ALGORITHM(fern::add, Array, Array)
    TRY_BINARY_ALGORITHM(fern::add, Array, Float)

    // Int, Long, Float, Array

    // TODO Error handling.
    assert(result);
    return result;
}


static PyMethodDef methods[] = {
    {"add", add, METH_VARARGS,
        "Bladiblah from numpy"},
    {nullptr, nullptr, 0, nullptr}
};

} // namespace fern


static fern::ThreadClient clien;

PyMODINIT_FUNC initfern_algorithm_numpy(
    void)
{
    PyObject* module = Py_InitModule("fern_algorithm_numpy", fern::methods);

    if(module) {
        // ...
    }
}
