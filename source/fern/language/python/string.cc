// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/python/string.h"


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
