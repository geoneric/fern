#include <Python.h>
#include "fern/algorithm/python_extension/gdal/error.h"


namespace fern {

void raise_runtime_error(
    std::string const& message)
{
    assert(!PyErr_Occurred());
    PyErr_SetString(PyExc_RuntimeError, message.c_str());
    assert(PyErr_Occurred());
}


void raise_unsupported_argument_type_exception(
    std::string const& type_representation)
{
    std::string message = "Unsupported argument type (" + type_representation +
        ")";
    raise_runtime_error(message);
}

} // namespace fern
