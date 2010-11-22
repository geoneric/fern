#include "Definition.h"

#include <cassert>



namespace ranally {
namespace language {

Definition::Definition(
  UnicodeString const& name)

  : _name(name)

{
  assert(!name.isEmpty());
}



Definition::Definition(
  Definition const& rhs)

  : _name(rhs._name)

{
}



Definition::~Definition()
{
}



UnicodeString const& Definition::name() const
{
  return _name;
}

} // namespace language
} // namespace ranally

