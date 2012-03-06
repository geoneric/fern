#include "Ranally/IO/PointFeature.h"
#include "Ranally/IO/PointAttribute.h"
#include "Ranally/IO/PointDomain.h"



namespace ranally {

PointFeature::PointFeature(
  PointDomainPtr const& domain)

  : Feature(domain->type()),
    _domain(domain)

{
}



PointFeature::~PointFeature()
{
}



PointDomain const& PointFeature::domain() const
{
  assert(_domain);
  return *_domain;
}

} // namespace ranally

