#include "ranally/python/exception.h"
#include "ranally/python/owned_reference.h"
#include "ranally/python/string.h"


namespace ranally {
namespace python {
namespace {

String format_error_message(
    PyObject* value,
    PyObject* /* traceback */)
{
    assert(value);
    assert(value != Py_None);
    OwnedReference value_description_object(PyObject_Str(value));
    assert(value_description_object);
    String valueDescription = as_unicode_string(value_description_object);

    return valueDescription;
}


String format_syntax_error_message(
    PyObject* value,
    PyObject* traceback)
{
    // TODO Unpack the filename, lineno, offset and text attributes and format
    //      a nice error message.
    return format_error_message(value, traceback);
}

} // Anonymous namespace


String error_message()
{
    assert(PyErr_Occurred());

    OwnedReference type;
    OwnedReference value;
    OwnedReference traceback;
    PyErr_Fetch(&type, &value, &traceback);
    // PyErr_NormalizeException(&type, &value, &traceback);
    PyErr_Clear();

    assert(!PyErr_Occurred());
    assert(type);

    // OwnedReference typeNameObject = PyObject_GetAttrString(type, "__name__");
    // assert(typeNameObject);
    // String typeName = as_unicode_string(typeNameObject);

    String message;

    if(PyErr_GivenExceptionMatches(type, PyExc_SyntaxError)) {
        message = format_syntax_error_message(value, traceback);
    }
    else {
        message = format_error_message(value, traceback);
    }

    assert(!message.is_empty());
    assert(!PyErr_Occurred());

    return message;
}

} // namespace python
} // namespace ranally
