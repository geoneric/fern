// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <memory>
#include "fern/language/feature/core/attribute.h"
#include "fern/language/feature/core/constant_value.h"


namespace fern {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
template<
    typename Value>
class ConstantAttribute:
    public Attribute
{

public:

    LOKI_DEFINE_CONST_VISITABLE()

                   ConstantAttribute   ();

                   ConstantAttribute   (Value const& value);

                   ConstantAttribute   (ConstantAttribute const&)=delete;

    ConstantAttribute& operator=       (ConstantAttribute const&)=delete;

                   ConstantAttribute   (ConstantAttribute&&)=delete;

    ConstantAttribute& operator=       (ConstantAttribute&&)=delete;

                   ~ConstantAttribute  ()=default;

    void           set                 (Value const& value);

    ConstantValue<Value> const& values () const;

private:

    std::unique_ptr<ConstantValue<Value>> _values;

};


template<
    typename Value>
inline ConstantAttribute<Value>::ConstantAttribute()

    : _values(std::make_unique<ConstantValue<Value>>())

{
}


template<
    typename Value>
inline ConstantAttribute<Value>::ConstantAttribute(
    Value const& value)

    : _values(std::make_unique<ConstantValue<Value>>(value))

{
}


template<
    typename Value>
void ConstantAttribute<Value>::set(
    Value const& value)
{
    _values->set(value);
}


template<
    typename Value>
ConstantValue<Value> const& ConstantAttribute<Value>::values() const
{
    return *_values;
}

} // namespace fern
