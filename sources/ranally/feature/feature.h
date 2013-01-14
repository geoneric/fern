#pragma once
#include <map>
#include <memory>
#include "ranally/core/string.h"


namespace ranally {

class Attribute;

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class Feature
{

    friend class featureTest;

public:

                   Feature             ()=default;

                   Feature             (Feature const&)=delete;

    Feature&       operator=           (Feature const&)=delete;

                   Feature             (Feature&&)=delete;

    Feature&       operator=           (Feature&&)=delete;

                   ~Feature            ()=default;

    template<class Attribute>
    void           add_attribute       (std::shared_ptr<Attribute>& attribute);

    size_t         nr_attributes       () const;

protected:

private:

    std::map<String, std::shared_ptr<Attribute>> _attributes;

};


//! Add an attribute to the feature.
/*!
  \tparam    Attribute Attribute specialization.
  \param     attribute Pointer to Attribute instance.
  \warning   An equally named attribute must not already exist in the feature.
*/
template<class Attribute>
inline void Feature::add_attribute(
    std::shared_ptr<Attribute>& attribute)
{
    assert(_attributes.find(attribute->name()) == _attributes.end());
    _attributes[attribute->name()] = attribute;
}

} // namespace ranally
