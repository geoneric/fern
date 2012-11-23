#include "Ranally/IO/polygon_attribute.h"
#include "Ranally/IO/polygon_feature.h"
#include "Ranally/IO/polygon_value.h"


namespace ranally {

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

} // namespace ranally
