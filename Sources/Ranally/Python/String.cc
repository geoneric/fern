#include "Ranally/Python/String.h"



namespace ranally {
namespace python {

String asUnicodeString(
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
} // namespace ranally

