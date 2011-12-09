#include "Ranally/Operation/Operation.h"



namespace ranally {
namespace operation {

Operation::Operation(
  UnicodeString const& name)

  : _name(name)

{
  assert(!name.isEmpty());
}



Operation::~Operation()
{
}



UnicodeString const& Operation::name() const
{
  return _name;
}

} // namespace operation
} // namespace ranally

