#pragma once
#include <cassert>
#include <memory>
#include "ranally/feature/attribute.h"
#include "ranally/feature/scalar_domain.h"
#include "ranally/feature/scalar_value.h"


namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
template<class T>
class ScalarAttribute:
    public Attribute
{

    friend class ScalarAttributeTest;

public:

                   ScalarAttribute     (String const& name,
                                        std::shared_ptr<ScalarDomain> const&
                                            domain,
                                        std::shared_ptr<ScalarValue<T>> const&
                                            value);

                   ScalarAttribute     (ScalarAttribute const&)=delete;

    ScalarAttribute& operator=         (ScalarAttribute const&)=delete;

                   ScalarAttribute     (ScalarAttribute&&)=delete;

    ScalarAttribute& operator=         (ScalarAttribute&&)=delete;

                   ~ScalarAttribute    ()=default;

protected:

private:

    std::shared_ptr<ScalarDomain> _domain;

    std::shared_ptr<ScalarValue<T>> _value;

};


template<class T>
inline ScalarAttribute<T>::ScalarAttribute(
    String const& name,
    std::shared_ptr<ScalarDomain> const& domain,
    std::shared_ptr<ScalarValue<T>> const& value)

    : Attribute(name),
      _domain(domain),
      _value(value)

{
    assert(_domain);
    assert(_value);
}

} // namespace ranally