#pragma once
#include <Python.h>
#include "ranally/core/string.h"


namespace ranally {
namespace python {

String             as_unicode_string   (PyObject* object);

} // namespace python
} // namespace ranally
