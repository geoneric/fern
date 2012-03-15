#include "Ranally/IO/Feature.h"



namespace ranally {

Feature::Feature(
  UnicodeString const& name,
  Domain::Type domainType)

  : _name(name),
    _domainType(domainType)

{
}



Feature::~Feature()
{
}



UnicodeString const& Feature::name() const
{
  return _name;
}



Domain::Type Feature::domainType() const
{
  return _domainType;
}

} // namespace ranally

