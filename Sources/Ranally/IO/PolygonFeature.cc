#include "Ranally/IO/PolygonFeature.h"
#include "Ranally/IO/PolygonAttribute.h"
#include "Ranally/IO/PolygonDomain.h"



namespace ranally {

PolygonFeature::PolygonFeature(
  String const& name,
  PolygonDomainPtr const& domain)

  : Feature(name, domain->type()),
    _domain(domain),
    _attributes()

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



PolygonAttributes const& PolygonFeature::attributes() const
{
  return _attributes;
}

} // namespace ranally

