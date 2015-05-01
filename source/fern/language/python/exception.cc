// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/python/exception.h"
#include "fern/core/exception.h"
#include "fern/core/string.h"
#include "fern/language/python/owned_reference.h"
#include "fern/language/python/string.h"


namespace fern {
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
    std::string value_description = as_unicode_string(value_description_object);

    BOOST_THROW_EXCEPTION(detail::ParseError()
        << detail::ExceptionMessage(value_description)
    );
}


void throw_syntax_error(
    std::string const& source_name,
    PyObject* value,
    PyObject* /* traceback */)
{
    assert(value);

    // (<error message>, (<source name>, <line>, <col>, <statement>))
    assert(PySequence_Check(value));
    assert(PySequence_Length(value) == 2);

    OwnedReference message_object(PySequence_GetItem(value, 0));
    std::string message = as_unicode_string(message_object);

    OwnedReference details_object(PySequence_GetItem(value, 1));
    // (<source name>, <line>, <col>, <statement>)
    assert(PySequence_Check(details_object));
    assert(PySequence_Length(details_object) == 4);

    // OwnedReference source_object(PySequence_GetItem(details_object, 0));
    // std::string source_name = as_unicode_string(source_object);

    OwnedReference line_nr_object(PySequence_GetItem(details_object, 1));
    assert(PyInt_Check((PyObject*)line_nr_object));
    long line_nr = PyInt_AsLong(line_nr_object);
    assert(!PyErr_Occurred());

    OwnedReference col_nr_object(PySequence_GetItem(details_object, 2));
    assert(PyInt_Check((PyObject*)col_nr_object));
    long col_nr = PyInt_AsLong(col_nr_object);
    assert(!PyErr_Occurred());

    OwnedReference statement_object(PySequence_GetItem(details_object, 3));
    std::string statement = as_unicode_string(statement_object);
    statement = strip(statement, "\n");

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
    std::string const& source_name)
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
    // std::string typeName = as_unicode_string(typeNameObject);

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
} // namespace fern
