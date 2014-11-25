#pragma once
#include <Python.h>
#include <numpy/arrayobject.h>
#include "fern/core/types.h"


namespace fern {
namespace python {
namespace numpy {

PyArrayObject*     add                 (float64_t value,
                                        PyArrayObject* array);

PyArrayObject*     add                 (PyArrayObject* array,
                                        float64_t value);

PyArrayObject*     add                 (PyArrayObject* array1,
                                        PyArrayObject* array2);

} // namespace numpy
} // namespace python
} // namespace fern
