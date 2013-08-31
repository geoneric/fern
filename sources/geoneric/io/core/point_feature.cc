#include "geoneric/io/point_feature.h"
#include "geoneric/io/point_attribute.h"
#include "geoneric/io/point_domain.h"


namespace geoneric {

PointFeature::PointFeature(
    String const& name,
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


bool PointFeature::exists(
    String const& name) const
{
    for(auto attribute: _attributes) {
        if(attribute->name() == name) {
            return true;
        }
    }
    return false;
}


void PointFeature::add(
    PointAttributePtr const& attribute)
{
    assert(!exists(attribute->name()));
    _attributes.push_back(attribute);
}

} // namespace geoneric
