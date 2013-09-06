#include "geoneric/io/core/polygon_feature.h"
#include "geoneric/io/core/polygon_attribute.h"
#include "geoneric/io/core/polygon_domain.h"


namespace geoneric {

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

} // namespace geoneric
