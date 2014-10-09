#include "fern/io/core/polygon_feature.h"
#include "fern/io/core/polygon_attribute.h"
#include "fern/io/core/polygon_domain.h"


namespace fern {

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

} // namespace fern
