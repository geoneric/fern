#include "ranally/io/feature.h"


namespace ranally {

Feature::Feature(
    String const& name,
    Domain::Type domainType)

    : _name(name),
      _domainType(domainType)

{
}


Feature::~Feature()
{
}


String const& Feature::name() const
{
    return _name;
}


Domain::Type Feature::domainType() const
{
    return _domainType;
}

} // namespace ranally
