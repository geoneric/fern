#pragma once
#include <Python.h>
#include "fern/core/string.h"


namespace fern {
namespace python {

bool               is_python_float     (PyObject* object);

double             python_float        (PyObject const* object);

PyObject*          python_object       (double value);

} // namespace python
} // namespace fern
