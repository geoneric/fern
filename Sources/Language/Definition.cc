#include "Definition.h"

#include <cassert>



namespace ranally {

Definition::Definition(
  UnicodeString const& name)

  : _name(name)

{
  assert(!name.isEmpty());
}



Definition::~Definition()
{
}



UnicodeString const& Definition::name() const
{
  return _name;
}

} // namespace ranally

