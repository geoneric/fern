// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <Python.h>
#include <string>
#include "fern/python/core/swig_runtime.h"


namespace fern {
namespace python {

SwigPyObject*      swig_object         (PyObject* object);

SwigPyObject*      swig_object         (PyObject* object,
                                        std::string const& typename_);

} // namespace python
} // namespace fern
