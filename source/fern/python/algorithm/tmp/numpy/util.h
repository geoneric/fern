#pragma once
#include <Python.h>
#include <numpy/arrayobject.h>
#include "fern/core/string.h"


namespace fern {
namespace python {

bool               is_numpy_array      (PyObject* object);

bool               is_masked_numpy_array(
                                        PyObject* object);

bool               is_masked_numpy_array(
                                        PyArrayObject* array_object);

PyObject*          mask_object         (PyObject* object);

PyObject*          mask_object         (PyArrayObject* array_object);

PyArrayObject*     numpy_array         (PyObject* object);

PyObject*          python_object       (PyArrayObject* array_object);

} // namespace python
} // namespace fern
