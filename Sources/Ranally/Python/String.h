#ifndef INCLUDED_RANALLY_PYTHON_STRING
#define INCLUDED_RANALLY_PYTHON_STRING

#include <Python.h>
#include <unicode/unistr.h>



namespace ranally {
namespace python {

UnicodeString      asUnicodeString     (PyObject* object);

} // namespace python
} // namespace ranally

#endif
