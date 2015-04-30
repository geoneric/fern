// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/feature/core/feature.h"


namespace fern {
namespace language {

//! Return the number of child-features of this feature.
/*!
  \sa        nr_attributes()
*/
size_t Feature::nr_features() const
{
    return _features.size();
}


size_t Feature::nr_features(
    Path const& path) const
{
    assert(contains_feature(path));

    return feature(path)->nr_features();
}


//! Return the names of the child-features of this feature.
/*!
  \sa        attribute_names()
*/
std::vector<String> Feature::feature_names() const
{
    std::vector<String> result;
    result.reserve(_features.size());

    for(auto pair: _features) {
        result.emplace_back(pair.first);
    }

    return result;
}


//! Return whether this feature contains a child-feature addressed by \a path.
/*!
  \param     path Path to feature.
  \sa        contains_attribute(Path const&)
*/
bool Feature::contains_feature(
    Path const& path) const
{
    return path.parent_path().generic_string().empty()
        ? _features.find(path.generic_string()) != _features.end()
        : static_cast<bool>(feature(path.generic_string()))
        ;
}


//! Add \a feature as a child-feature addressed by \a path.
/*!
  \param     path Path to store feature at. The parent feature must exist.
  \param     feature Feature to store.
  \warning   There must not be an attribute stored at \a path already.
  \sa        add_attribute(Path const&, std::shared_ptr<Attribute> const&)
*/
void Feature::add_feature(
    Path const& path,
    std::shared_ptr<Feature> const& feature)
{
    assert(!contains_attribute(path));

    Feature* parent_feature = this->parent_feature(path);

#ifndef NDEBUG
    auto result =
#endif
    parent_feature->_features.insert(std::make_pair(
        path.filename().generic_string(), feature));

    assert(result.second);
}


//! Return feature addressed by \a path.
/*!
  \param     path Path to feature.

  If no such feature exists, a default constructed shared pointer is returned.
*/
std::shared_ptr<Feature> Feature::feature(
    Path const& path) const
{
    return feature(path.names());
}


std::shared_ptr<Feature> Feature::feature(
    std::vector<std::string> names) const
{
    std::shared_ptr<Feature> result;

    if(!names.empty()) {
        if(_features.find(names[0]) != _features.end()) {
            result = _features.at(names[0]);

            if(names.size() > 1) {
                // The caller wants the child-feature. Recurse with the tail
                // of the names collection.
                result = result->feature(std::vector<std::string>(
                    names.begin() + 1, names.end()));
            }
        }
    }

    return result;
}


//! Return the number of attributes of this feature.
/*!
  \sa        nr_features()
*/
size_t Feature::nr_attributes() const
{
    return _attributes.size();
}


size_t Feature::nr_attributes(
    Path const& path) const
{
    return parent_feature(path)->nr_attributes();
}


//! Return the names of the attributes of this feature.
/*!
  \sa        feature_names()
*/
std::vector<String> Feature::attribute_names() const
{
    std::vector<String> result;
    result.reserve(_attributes.size());

    for(auto pair: _attributes) {
        result.emplace_back(pair.first);
    }

    return result;
}


//! Return whether this feature contains an attribute addressed by \a path.
/*!
  \param     path Path to attribute.
  \sa        contains_feature(Path const&)
*/
bool Feature::contains_attribute(
    Path const& path) const
{
    return path.parent_path().generic_string().empty()
        ? _attributes.find(path.generic_string()) != _attributes.end()
        : static_cast<bool>(attribute(path.generic_string()))
        ;
}


//! Add \a attribute as an attribute addressed by \a path.
/*!
  \param     path Path to store attribute at. The parent feature must exist.
  \param     attribute attribute to store.
  \warning   There must not be a feature stored at \a path already.
  \sa        add_feature(Path const&, std::shared_ptr<Attribute> const&)
*/
void Feature::add_attribute(
    Path const& path,
    std::shared_ptr<Attribute> const& attribute)
{
    assert(!contains_feature(path));

    Feature* parent_feature = this->parent_feature(path);

#ifndef NDEBUG
    auto result =
#endif
    parent_feature->_attributes.insert(std::make_pair(
        path.filename().generic_string(), attribute));

    assert(result.second);
}


//! Return attribute addressed by \a path.
/*!
  \param     path Path to attribute. The attribute must exist.
*/
std::shared_ptr<Attribute> Feature::attribute(
    Path const& path) const
{
    return parent_feature(path)->_attributes.at(
        path.filename().generic_string());
}


Feature* Feature::parent_feature(
    Path const& path)
{
    assert(path.parent_path().generic_string().empty() ||
        contains_feature(path.parent_path()));
    return path.parent_path().generic_string().empty()
        ? this : feature(path.parent_path()).get();
}


Feature const* Feature::parent_feature(
    Path const& path) const
{
    assert(path.parent_path().generic_string().empty() ||
        contains_feature(path.parent_path()));
    return path.parent_path().generic_string().empty()
        ? this : feature(path.parent_path()).get();
}

} // namespace language
} // namespace fern
