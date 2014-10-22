#pragma once
#include <Python.h>
#include <numpy/arrayobject.h>


namespace fern {
namespace python {

PyArrayObject*     sqrt                (PyArrayObject* array);

} // namespace python
} // namespace fern
