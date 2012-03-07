#include "Ranally/IO/PointAttribute.h"
#include "Ranally/IO/PointFeature.h"
#include "Ranally/IO/PointValue.h"



namespace ranally {

PointAttribute::PointAttribute(
  PointDomainPtr const& domain)

  : Attribute(),
    _domain(domain),
    _feature(),
    _values()

{
  assert(domain);
}



PointAttribute::~PointAttribute()
{
}



PointDomainPtr const& PointAttribute::domain() const
{
  return _domain;
}



PointFeaturePtr const& PointAttribute::feature() const
{
  return _feature;
}



PointValues const& PointAttribute::values() const
{
  return _values;
}

} // namespace ranally

