#include "Ranally/IO/polygon_feature.h"
#include "Ranally/IO/polygon_attribute.h"
#include "Ranally/IO/polygon_domain.h"


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
