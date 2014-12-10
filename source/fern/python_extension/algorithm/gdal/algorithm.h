#pragma once
#include <Python.h>


namespace fern {
namespace python {
namespace gdal {

PyObject*          add                 (PyObject* self,
                                        PyObject* arguments);

PyObject*          slope               (PyObject* self,
                                        PyObject* arguments);

} // namespace gdal
} // namespace python
} // namespace fern
