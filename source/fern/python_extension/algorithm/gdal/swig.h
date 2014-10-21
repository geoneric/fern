#pragma once
#include <Python.h>
#include "fern/algorithm/python_extension/swig_runtime.h"
#include "fern/core/string.h"


namespace fern {
namespace python {

SwigPyObject*      swig_object         (PyObject* object);

SwigPyObject*      swig_object         (PyObject* object,
                                        String const& typename_);

} // namespace python
} // namespace fern
