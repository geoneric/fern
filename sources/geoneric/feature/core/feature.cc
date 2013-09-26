#include "geoneric/feature/core/feature.h"


namespace geoneric {

void Feature::add_attribute(
    String const& name,
    std::shared_ptr<Attribute> const& attribute)
{
    auto result = insert(std::make_pair(name, attribute));
    assert(result.second);
}


std::shared_ptr<Attribute> const& Feature::attribute(
    String const& name)
{
    assert(contains_attribute(name));
    return at(name);
}


size_t Feature::nr_attributes() const
{
    return size();
}


bool Feature::contains_attribute(
    String const& name)
{
    return find(name) != end();
}


std::vector<String> Feature::attribute_names() const
{
    std::vector<String> result;
    result.reserve(size());

    for(auto pair: *this) {
        result.push_back(pair.first);
    }

    return result;
}

} // namespace geoneric
