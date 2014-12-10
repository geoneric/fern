#pragma once
#include <Python.h>


namespace fern {
namespace python {
namespace numpy {

PyObject*          add                 (PyObject* self,
                                        PyObject* arguments);

PyObject*          sqrt                (PyObject* self,
                                        PyObject* arguments);

} // namespace numpy
} // namespace python
} // namespace fern
