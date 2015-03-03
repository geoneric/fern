#include "fern/python/string.h"


namespace fern {
namespace python {

String as_unicode_string(
    PyObject* object)
{
    // TODO What do we know about the encoding of the input string? Use it to
    //      create the correct UnicodeString instance. Currently UTF8 is assumed
    //      by String's constructor.
    char* bytes = PyString_AsString(object);
    assert(bytes);
    return String(bytes);
}

} // namespace python
} // namespace fern
