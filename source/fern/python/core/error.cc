// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include <Python.h>
#include "fern/python/core/error.h"


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
