#pragma once
#include <cassert>
#include <memory>
#include "fern/operation/core/type_traits.h"
#include "fern/feature/attribute.h"
#include "fern/feature/feature_domain.h"
#include "fern/feature/domain_value.h"


namespace fern {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
template<
    class Model,
    class T>
class DomainAttribute:
    public Attribute
{

public:

                   DomainAttribute     (
                                   std::shared_ptr<FeatureDomain<Model>> const&
                                        domain,
                                   std::shared_ptr<DomainValue<T>> const&
                                       value);

                   DomainAttribute     (DomainAttribute const&)=delete;

    DomainAttribute& operator=         (DomainAttribute const&)=delete;

                   DomainAttribute     (DomainAttribute&&)=delete;

    DomainAttribute& operator=         (DomainAttribute&&)=delete;

                   ~DomainAttribute    ()=default;

    std::shared_ptr<DomainValue<T>> const& value() const;

private:

    std::shared_ptr<FeatureDomain<Model>> _domain;

    std::shared_ptr<DomainValue<T>> _value;

};


template<
    class Model,
    class T>
inline DomainAttribute<Model, T>::DomainAttribute(
    std::shared_ptr<FeatureDomain<Model>> const& domain,
    std::shared_ptr<DomainValue<T>> const& value)

    // TODO Get data type from model type traits.
    : Attribute(DT_POINT, TypeTraits<T>::value_type),
      _domain(domain),
      _value(value)

{
    assert(_domain);
    assert(_value);
    assert(_domain->size() == _value->size());
}


template<
    class Model,
    class T>
inline std::shared_ptr<DomainValue<T>> const&
    DomainAttribute<Model, T>::value() const
{
    return _value;
}

} // namespace fern
