#pragma once
#include <Python.h>
#include <numpy/arrayobject.h>


namespace fern {
namespace python {
namespace numpy {

PyArrayObject*     sqrt                (PyArrayObject* array);

} // namespace numpy
} // namespace python
} // namespace fern
