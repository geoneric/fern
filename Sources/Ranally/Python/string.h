#pragma once
#include <Python.h>
#include "Ranally/Util/string.h"


namespace ranally {
namespace python {

String             asUnicodeString     (PyObject* object);

} // namespace python
} // namespace ranally
