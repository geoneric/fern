#pragma once
#include <Python.h>
#include "fern/python_extension/algorithm/core/binary_operation_map.h"


namespace fern {
namespace python {

PyObject*          add                 (PyObject* self,
                                        PyObject* arguments);

PyObject*          sqrt                (PyObject* self,
                                        PyObject* arguments);

} // namespace python
} // namespace fern
