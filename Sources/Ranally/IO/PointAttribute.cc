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

} // namespace ranally

