#pragma once
#include <Python.h>
#include <numpy/arrayobject.h>
#include "fern/core/string.h"


namespace fern {
namespace python {

bool               is_numpy_array      (PyObject* object);

PyArrayObject*     numpy_array         (PyObject* object);

PyObject*          python_object       (PyArrayObject* array_object);

} // namespace python
} // namespace fern
