#include "Ranally/IO/PolygonAttribute.h"
#include "Ranally/IO/PolygonFeature.h"
#include "Ranally/IO/PolygonValue.h"



namespace ranally {

PolygonAttribute::PolygonAttribute(
  PolygonDomainPtr const& domain)

  : Attribute(),
    _domain(domain),
    _feature(),
    _values()

{
}



PolygonAttribute::~PolygonAttribute()
{
}

} // namespace ranally

