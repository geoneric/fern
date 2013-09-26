#pragma once
#include <map>
#include <memory>
#include "geoneric/core/string.h"
#include "geoneric/feature/core/attribute.h"


namespace geoneric {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class Feature:
    private std::map<String, std::shared_ptr<Attribute>>
{

public:

                   Feature             ()=default;

                   Feature             (Feature const&)=delete;

    Feature&       operator=           (Feature const&)=delete;

                   Feature             (Feature&&)=delete;

    Feature&       operator=           (Feature&&)=delete;

                   ~Feature            ()=default;

    size_t         nr_attributes       () const;

    bool           contains_attribute  (String const& name);

    void           add_attribute       (String const& name,
                                        std::shared_ptr<Attribute> const& attribute);

    std::vector<String> attribute_names  () const;

    std::shared_ptr<Attribute> const& attribute(
                                        String const& name);

private:

};

} // namespace geoneric
