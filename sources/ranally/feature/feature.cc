#include "ranally/feature/feature.h"


namespace ranally {

// Feature::Feature()
// {
// }
// 
// 
// Feature::~Feature()
// {
// }

std::shared_ptr<Attribute> Feature::attribute(
         String const& name) const
{
    assert(_attributes.find(name) != _attributes.end());
    return _attributes.find(name)->second;
}


size_t Feature::nr_attributes() const
{
    return _attributes.size();
}

} // namespace ranally
