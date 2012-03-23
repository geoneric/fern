#include "Ranally/IO/PointFeature.h"
#include <boost/foreach.hpp>
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



bool PointFeature::exists(
  UnicodeString const& name) const
{
  BOOST_FOREACH(PointAttributePtr const& attribute, _attributes) {
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

} // namespace ranally

