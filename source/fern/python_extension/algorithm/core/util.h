#pragma once
#include <Python.h>
#include "fern/core/string.h"
#include "fern/core/types.h"


namespace fern {
namespace python {

bool               is_python_bool      (PyObject* object);

bool               is_python_int       (PyObject* object);

bool               is_python_float     (PyObject* object);

bool               python_bool         (PyObject const* object);

int64_t            python_int          (PyObject const* object);

float64_t          python_float        (PyObject const* object);

PyObject*          python_object       (int64_t value);

PyObject*          python_object       (float64_t value);

} // namespace python
} // namespace fern
