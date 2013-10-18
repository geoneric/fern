#include "geoneric/feature/core/feature.h"


namespace geoneric {

size_t Feature::nr_features() const
{
    return _features.size();
}


size_t Feature::nr_features(
    Path const& feature_path) const
{
    assert(contains_feature(feature_path));
    return feature(feature_path)->nr_features();
}


std::vector<String> Feature::feature_names() const
{
    std::vector<String> result;
    result.reserve(_features.size());

    for(auto pair: _features) {
        result.push_back(pair.first);
    }

    return result;
}


bool Feature::contains_feature(
    Path const& path) const
{
    return static_cast<bool>(feature(path));
}


void Feature::add_feature(
    Path const& path,
    std::shared_ptr<Feature> const& feature)
{
    assert(contains_feature(path.parent_path()));
    std::shared_ptr<Feature> parent_feature(this->feature(path.parent_path()));
    auto result = parent_feature->_features.insert(std::make_pair(
        String(path.filename()), feature));
    assert(result.second);
}


std::shared_ptr<Feature> Feature::feature(
    Path const& path) const
{
    return feature(path.names());
}


std::shared_ptr<Feature> Feature::feature(
    std::vector<String> names) const
{
    std::shared_ptr<Feature> result;

    if(!names.empty()) {
        if(_features.find(names[0]) != _features.end()) {
            result = _features.at(names[0]);

            if(names.size() > 1) {
                // The caller wants the child-feature. Recurse with the tail
                // of the names collection.
                result = result->feature(std::vector<String>(names.begin() + 1,
                    names.end()));
            }
        }
    }

    return result;
}


size_t Feature::nr_attributes() const
{
    return _attributes.size();
}


size_t Feature::nr_attributes(
    Path const& feature_path) const
{
    assert(contains_feature(feature_path));
    return feature(feature_path)->nr_attributes();
}


std::vector<String> Feature::attribute_names() const
{
    std::vector<String> result;
    result.reserve(_attributes.size());

    for(auto pair: _attributes) {
        result.push_back(pair.first);
    }

    return result;
}


bool Feature::contains_attribute(
    Path const& attribute_path) const
{
    return static_cast<bool>(attribute(attribute_path));
}


void Feature::add_attribute(
    Path const& path,
    std::shared_ptr<Attribute> const& attribute)
{
    assert(contains_feature(path.parent_path()));
    std::shared_ptr<Feature> parent_feature(this->feature(path.parent_path()));
    auto result = parent_feature->_attributes.insert(std::make_pair(
        String(path.filename()), attribute));
    assert(result.second);
}


std::shared_ptr<Attribute> Feature::attribute(
    Path const& path) const
{
    std::shared_ptr<Attribute> result;
    std::shared_ptr<Feature> feature(this->feature(path.parent_path()));

    if(feature && feature->contains_attribute(path.filename())) {
        result = feature->attribute(path.filename());
    }

    return result;
}

} // namespace geoneric
