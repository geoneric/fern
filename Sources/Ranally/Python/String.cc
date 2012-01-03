#include "Ranally/Python/String.h"



namespace ranally {
namespace python {

UnicodeString asUnicodeString(
  PyObject* object)
{
  char* bytes = PyString_AsString(object);
  assert(bytes);
  return UnicodeString(bytes);
}

} // namespace python
} // namespace ranally

