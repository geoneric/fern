// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/python/core/swig.h"


namespace fern {
namespace python {

SwigPyObject* swig_object(
    PyObject* object)
{
    return SWIG_Python_GetSwigThis(object);
}


SwigPyObject* swig_object(
    PyObject* object,
    String const& typename_)
{
    SwigPyObject* result = swig_object(object);

    if(result != nullptr && String(result->ty->name) == typename_) {
        return result;
    }

    return nullptr;
}

} // namespace python
} // namespace fern
