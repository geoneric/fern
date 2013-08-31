#pragma once
#include <map>
#include <memory>
#include "ranally/core/string.h"


namespace ranally {

class Attribute;
class Domain;

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class Feature
{

public:

    template<class Domain>
                   Feature             (std::shared_ptr<Domain> const& domain);

                   Feature             (Feature const&)=delete;

    Feature&       operator=           (Feature const&)=delete;

                   Feature             (Feature&&)=delete;

    Feature&       operator=           (Feature&&)=delete;

                   ~Feature            ()=default;

    template<class Attribute>
    void           add_attribute       (String const& name,
                                        std::shared_ptr<Attribute> const&
                                            attribute);

    template<class Attribute>
    std::shared_ptr<Attribute> attribute(
                                        String const& name) const;

    template<class Domain>
    std::shared_ptr<Domain> domain     () const;

    size_t         nr_attributes       () const;

private:

    std::shared_ptr<Domain> _domain;

    std::map<String, std::shared_ptr<Attribute>> _attributes;

    std::shared_ptr<Attribute> attribute(
                                        String const& name) const;

};


template<
    class Domain>
inline Feature::Feature(
    std::shared_ptr<Domain> const& domain)

    : _domain(domain)

{
}


template<
    class Domain>
std::shared_ptr<Domain> Feature::domain() const
{
    return std::dynamic_pointer_cast<Domain>(_domain);
}


//! Add an attribute to the feature.
/*!
  \tparam    Attribute Attribute specialization.
  \param     attribute Pointer to Attribute instance.
  \warning   An equally named attribute must not already exist in the feature.
*/
template<
    class Attribute>
inline void Feature::add_attribute(
    String const& name,
    std::shared_ptr<Attribute> const& attribute)
{
    assert(_attributes.find(name) == _attributes.end());
    _attributes[name] = attribute;
}


template<
    class Attribute>
inline std::shared_ptr<Attribute> Feature::attribute(
    String const& name) const
{
    return std::dynamic_pointer_cast<Attribute>(attribute(name));
}

} // namespace ranally
