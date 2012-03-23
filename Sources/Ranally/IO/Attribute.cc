#include "Ranally/IO/Attribute.h"
#include "Ranally/IO/Domain.h"
#include "Ranally/IO/Feature.h"
#include "Ranally/IO/Value.h"



namespace ranally {

Attribute::Attribute(
  UnicodeString const& name)

  : _name(name)

{
}



Attribute::~Attribute()
{
}



UnicodeString const& Attribute::name() const
{
  return _name;
}

} // namespace ranally

