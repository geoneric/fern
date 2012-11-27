#pragma once
#include <Python.h>
#include "ranally/core/string.h"


namespace ranally {
namespace python {

String             asUnicodeString     (PyObject* object);

} // namespace python
} // namespace ranally
