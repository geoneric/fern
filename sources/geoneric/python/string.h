#pragma once
#include <Python.h>
#include "geoneric/core/string.h"


namespace geoneric {
namespace python {

String             as_unicode_string   (PyObject* object);

} // namespace python
} // namespace geoneric
