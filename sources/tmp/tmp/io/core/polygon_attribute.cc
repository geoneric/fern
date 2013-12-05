#include "fern/io/core/polygon_attribute.h"
#include "fern/io/core/polygon_feature.h"
#include "fern/io/core/polygon_value.h"


namespace fern {

PolygonAttribute::PolygonAttribute(
    String const& name,
    PolygonDomainPtr const& domain)

    : Attribute(name),
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

} // namespace fern