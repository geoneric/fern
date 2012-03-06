#include "Ranally/IO/PolygonFeature.h"
#include "Ranally/IO/PolygonAttribute.h"
#include "Ranally/IO/PolygonDomain.h"



namespace ranally {

PolygonFeature::PolygonFeature(
  PolygonDomainPtr const& domain)

  : Feature(domain->type()),
    _domain(domain)

{
}



PolygonFeature::~PolygonFeature()
{
}



PolygonDomain const& PolygonFeature::domain() const
{
  assert(_domain);
  return *_domain;
}

} // namespace ranally

