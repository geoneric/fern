#include "Ranally/IO/Feature.h"



namespace ranally {

Feature::Feature(
  Domain::Type domainType)

  : _domainType(domainType)

{
}



Feature::~Feature()
{
}



Domain::Type Feature::domainType() const
{
  return _domainType;
}

} // namespace ranally

