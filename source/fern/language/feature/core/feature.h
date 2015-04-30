// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <map>
#include <memory>
#include "fern/core/path.h"
#include "fern/core/string.h"
#include "fern/language/feature/core/attribute.h"


namespace fern {
namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class Feature
{

public:

                   Feature             ()=default;

                   Feature             (Feature const&)=delete;

    Feature&       operator=           (Feature const&)=delete;

                   Feature             (Feature&&)=delete;

    Feature&       operator=           (Feature&&)=delete;

                   ~Feature            ()=default;

    size_t         nr_features         () const;

    size_t         nr_features         (Path const& path) const;

    std::vector<String> feature_names  () const;

    bool           contains_feature    (Path const& path) const;

    void           add_feature         (Path const& path,
                                        std::shared_ptr<Feature> const&
                                            feature);

    std::shared_ptr<Feature> feature   (Path const& path) const;

    size_t         nr_attributes       () const;

    size_t         nr_attributes       (Path const& path) const;

    std::vector<String> attribute_names() const;

    bool           contains_attribute  (Path const& path) const;

    void           add_attribute       (Path const& path,
                                        std::shared_ptr<Attribute> const&
                                            attribute);

    std::shared_ptr<Attribute> attribute(
                                        Path const& path) const;

private:

    std::map<String, std::shared_ptr<Feature>> _features;

    std::map<String, std::shared_ptr<Attribute>> _attributes;

    std::shared_ptr<Feature> feature   (std::vector<std::string> names) const;

    Feature const* parent_feature      (Path const& path) const;

    Feature*       parent_feature      (Path const& path);

};

} // namespace language
} // namespace fern
