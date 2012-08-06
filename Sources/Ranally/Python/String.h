#ifndef INCLUDED_RANALLY_PYTHON_STRING
#define INCLUDED_RANALLY_PYTHON_STRING

#include <Python.h>
#include "Ranally/Util/String.h"



namespace ranally {
namespace python {

String             asUnicodeString     (PyObject* object);

} // namespace python
} // namespace ranally

#endif
