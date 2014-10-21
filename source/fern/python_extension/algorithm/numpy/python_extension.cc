#include <Python.h>
#include "fern/core/thread_client.h"
#include "fern/python_extension/core/error.h"
#include "fern/python_extension/algorithm/numpy/add_overloads.h"


namespace fern {
namespace python {

static PyObject* add(
    PyObject* /* self */,
    PyObject* arguments)
{
    /// init_numpy();

    PyObject* value1_object;
    PyObject* value2_object;

    if(!PyArg_ParseTuple(arguments, "OO", &value1_object, &value2_object)) {
        return nullptr;
    }

    PyObject* result{nullptr};

    try {
        BinaryAlgorithmKey key(data_type(value1_object),
            data_type(value2_object));

        if(add_overloads.find(key) == add_overloads.end()) {
            raise_unsupported_overload_exception(value1_object, value2_object);
            result = nullptr;
        }
        else {
            result = add_overloads[key](value1_object, value2_object);
        }

        assert((PyErr_Occurred() && result == nullptr) ||
            (!PyErr_Occurred() && result != nullptr));
    }
    catch(std::runtime_error const& exception) {
        PyErr_SetString(PyExc_RuntimeError, exception.what());
        assert(result == nullptr);
    }
    catch(std::exception const& exception) {
        PyErr_SetString(PyExc_StandardError, exception.what());
        assert(result == nullptr);
    }

    assert((PyErr_Occurred() && result == nullptr) ||
        (!PyErr_Occurred() && result != nullptr));
    assert(result != Py_None);
    return result;
}


static PyMethodDef methods[] = {
    {"add", add, METH_VARARGS, "Add two arguments and return the result"},
    {nullptr, nullptr, 0, nullptr}
};

} // namespace python
} // namespace fern


static fern::ThreadClient client;


PyMODINIT_FUNC init_fern_algorithm_numpy(
    void)
{
    PyObject* module = Py_InitModule("_fern_algorithm_numpy",
        fern::python::methods);

    if(module) {
        // ...
    }
}
