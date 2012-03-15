#include "Ranally/IO/PointFeature.h"
#include "Ranally/IO/PointAttribute.h"
#include "Ranally/IO/PointDomain.h"



namespace ranally {

PointFeature::PointFeature(
  UnicodeString const& name,
  PointDomainPtr const& domain)

  : Feature(name, domain->type()),
    _domain(domain),
    _attributes()

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



PointAttributes const& PointFeature::attributes() const
{
  return _attributes;
}

} // namespace ranally

