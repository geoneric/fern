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
  assert(domain);
}



PolygonAttribute::~PolygonAttribute()
{
}



PolygonDomainPtr const& PolygonAttribute::domain() const
{
  return _domain;
}



PolygonFeaturePtr const& PolygonAttribute::feature() const
{
  return _feature;
}



std::vector<PolygonValuePtr> PolygonAttribute::values() const
{
  return _values;
}

} // namespace ranally

