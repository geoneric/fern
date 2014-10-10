#pragma once
#include <Python.h>
#include <numpy/arrayobject.h>


namespace fern {

PyArrayObject*     add                 (PyArrayObject const* array_object1,
                                        PyArrayObject const* array_object2);

PyArrayObject*     add                 (PyArrayObject const* array_object,
                                        PyFloatObject const* float_object);

} // namespace fern
