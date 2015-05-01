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

std::string type_representation(
    PyObject* object)
{
    PyObject* type_object = reinterpret_cast<PyObject*>(object->ob_type);
    PyObject* representation_object = PyObject_Repr(type_object);

    return PyString_AsString(representation_object);
}

} // namespace detail


void raise_runtime_error(
    std::string const& message)
{
    assert(!PyErr_Occurred());
    // TODO utf8 or default encoding?
    PyErr_SetString(PyExc_RuntimeError, message.c_str());
    assert(PyErr_Occurred());
}


void raise_unsupported_argument_type_exception(
    std::string const& type_representation)
{
    std::string message = "Unsupported argument type (" +
        type_representation + ")";
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
    std::string const& type_representation)
{
    std::string message = "Unsupported overload for argument type " +
        type_representation;
    raise_runtime_error(message);
}


void raise_unsupported_overload_exception(
    std::string const& type_representation1,
    std::string const& type_representation2)
{
    std::string message = "Unsupported overload for argument types " +
        type_representation1 + ", " + type_representation2;
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
