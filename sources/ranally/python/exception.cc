#include "ranally/python/exception.h"
#include "ranally/python/owned_reference.h"
#include "ranally/python/string.h"


namespace ranally {
namespace python {
namespace {

String formatErrorMessage(
    PyObject* value,
    PyObject* /* traceback */)
{
    assert(value);
    assert(value != Py_None);
    OwnedReference valueDescriptionObject(PyObject_Str(value));
    assert(valueDescriptionObject);
    String valueDescription = asUnicodeString(valueDescriptionObject);

    return valueDescription;
}


String formatSyntaxErrorMessage(
    PyObject* value,
    PyObject* traceback)
{
    // TODO Unpack the filename, lineno, offset and text attributes and format
    //      a nice error message.
    return formatErrorMessage(value, traceback);
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
    // String typeName = asUnicodeString(typeNameObject);

    String message;

    if(PyErr_GivenExceptionMatches(type, PyExc_SyntaxError)) {
        message = formatSyntaxErrorMessage(value, traceback);
    }
    else {
        message = formatErrorMessage(value, traceback);
    }

    assert(!message.isEmpty());
    assert(!PyErr_Occurred());

    return message;
}

} // namespace python
} // namespace ranally
