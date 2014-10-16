#include "fern/algorithm/python_extension/gdal/swig.h"


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
