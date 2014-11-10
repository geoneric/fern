#pragma once
#include <Python.h>


namespace fern {
namespace python {

PyObject*          add                 (PyObject* self,
                                        PyObject* arguments);

PyObject*          slope               (PyObject* self,
                                        PyObject* arguments);

} // namespace python
} // namespace fern
