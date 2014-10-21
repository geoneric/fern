#include <Python.h>
#include "fern/python_extension/core/error.h"


namespace fern {
namespace python {
namespace detail {

String type_representation(
    PyObject* object)
{
    PyObject* type_object = reinterpret_cast<PyObject*>(object->ob_type);
    PyObject* representation_object = PyObject_Repr(type_object);

    return PyString_AsString(representation_object);
}

} // namespace detail


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


void raise_unsupported_argument_type_exception(
    PyObject* object)
{
    PyObject* type_object = reinterpret_cast<PyObject*>(object->ob_type);
    PyObject* representation_object = PyObject_Repr(type_object);

    raise_unsupported_argument_type_exception(PyString_AsString(
        representation_object));
}


void raise_unsupported_overload_exception(
    String const& type_representation)
{
    String message = String("Unsupported overload for argument type ") +
        type_representation;
    raise_runtime_error(message);
}


void raise_unsupported_overload_exception(
    String const& type_representation1,
    String const& type_representation2)
{
    String message = String("Unsupported overload for argument types ") +
        type_representation1 + String(", ") + type_representation2;
    raise_runtime_error(message);
}


void raise_unsupported_overload_exception(
    PyObject* object)
{
    raise_unsupported_overload_exception(detail::type_representation(object));
}


void raise_unsupported_overload_exception(
    PyObject* object1,
    PyObject* object2)
{
    raise_unsupported_overload_exception(
        detail::type_representation(object1),
        detail::type_representation(object2));
}

} // namespace python
} // namespace fern
