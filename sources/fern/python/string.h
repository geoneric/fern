#pragma once
#include <Python.h>
#include "fern/core/string.h"


namespace fern {
namespace python {

String             as_unicode_string   (PyObject* object);

} // namespace python
} // namespace fern
