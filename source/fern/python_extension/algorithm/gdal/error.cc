#include <Python.h>
#include "fern/python_extension/algorithm/gdal/error.h"


namespace fern {
namespace python {

void raise_runtime_error(
    String const& message)
{
    assert(!PyErr_Occurred());
    // TODO utf8 or default encoding?
    PyErr_SetString(PyExc_RuntimeError, message.encode_in_utf8().c_str());
    assert(PyErr_Occurred());
}


void raise_unsupported_argument_type_exception(
    String const& type_representation)
{
    String message = String("Unsupported argument type (") +
        type_representation + String(")");
    raise_runtime_error(message);
}

} // namespace python
} // namespace fern
