#include "ranally/io/point_attribute.h"
#include "ranally/io/point_feature.h"
#include "ranally/io/point_value.h"


namespace ranally {

PointAttribute::PointAttribute(
    String const& name,
    PointDomainPtr const& domain)

    : Attribute(name),
      _domain(domain),
      _feature(),
      _values()

{
    assert(domain);
}


PointAttribute::~PointAttribute()
{
}


PointDomainPtr const& PointAttribute::domain() const
{
    return _domain;
}


PointFeaturePtr const& PointAttribute::feature() const
{
    return _feature;
}


PointValues const& PointAttribute::values() const
{
    return _values;
}

} // namespace ranally
