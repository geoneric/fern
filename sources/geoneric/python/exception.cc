#include "ranally/python/exception.h"
#include "ranally/core/exception.h"
#include "ranally/python/owned_reference.h"
#include "ranally/python/string.h"


namespace ranally {
namespace python {
namespace {

void throw_error(
    PyObject* value,
    PyObject* /* traceback */)
{
    assert(value);
    assert(value != Py_None);
    OwnedReference value_description_object(PyObject_Str(value));
    assert(value_description_object);
    String value_description = as_unicode_string(value_description_object);

    BOOST_THROW_EXCEPTION(detail::ParseError()
        << detail::ExceptionMessage(value_description)
    );
}


void throw_syntax_error(
    String const& source_name,
    PyObject* value,
    PyObject* /* traceback */)
{
    assert(value);

    // (<error message>, (<source name>, <line>, <col>, <statement>))
    assert(PySequence_Check(value));
    assert(PySequence_Length(value) == 2);

    OwnedReference message_object(PySequence_GetItem(value, 0));
    String message = as_unicode_string(message_object);

    OwnedReference details_object(PySequence_GetItem(value, 1));
    // (<source name>, <line>, <col>, <statement>)
    assert(PySequence_Check(details_object));
    assert(PySequence_Length(details_object) == 4);

    // OwnedReference source_object(PySequence_GetItem(details_object, 0));
    // String source_name = as_unicode_string(source_object);

    OwnedReference line_nr_object(PySequence_GetItem(details_object, 1));
    assert(PyInt_Check((PyObject*)line_nr_object));
    long line_nr = PyInt_AsLong(line_nr_object);
    assert(!PyErr_Occurred());

    OwnedReference col_nr_object(PySequence_GetItem(details_object, 2));
    assert(PyInt_Check((PyObject*)col_nr_object));
    long col_nr = PyInt_AsLong(col_nr_object);
    assert(!PyErr_Occurred());

    OwnedReference statement_object(PySequence_GetItem(details_object, 3));
    String statement = as_unicode_string(statement_object).strip("\n");

    assert(!PyErr_Occurred());

    detail::ParseError exception;

    exception
        << detail::ExceptionSourceName(source_name)
        << detail::ExceptionLineNr(line_nr)
        << detail::ExceptionColNr(col_nr)
        << detail::ExceptionStatement(statement)
        << detail::ExceptionMessage(message)
        ;

    BOOST_THROW_EXCEPTION(exception);
}

} // Anonymous namespace


void throw_exception(
    String const& source_name)
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

    if(PyErr_GivenExceptionMatches(type, PyExc_SyntaxError)) {
        throw_syntax_error(source_name, value, traceback);
    }
    else {
        assert(false);
        throw_error(value, traceback);
    }

    assert(!PyErr_Occurred());
}

} // namespace python
} // namespace ranally
