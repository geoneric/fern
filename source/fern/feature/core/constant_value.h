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
#include "fern/feature/core/values.h"


namespace fern {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
template<
    typename Value>
class ConstantValue:
    public Values
{

public:

                   ConstantValue       ()=default;

                   ConstantValue       (Value const& value);

                   ConstantValue       (ConstantValue const&)=delete;

    ConstantValue& operator=           (ConstantValue const&)=delete;

                   ConstantValue       (ConstantValue&&)=delete;

    ConstantValue& operator=           (ConstantValue&&)=delete;

                   ~ConstantValue      ()=default;

    void           set                 (Value const& value);

    Value const&   value               () const;

private:

    Value          _value;

};


template<
    typename Value>
inline ConstantValue<Value>::ConstantValue(
    Value const& value)

    : Values(),
      _value(value)

{
}


template<
    typename Value>
inline void ConstantValue<Value>::set(
    Value const& value)
{
    _value = value;
}


template<
    typename Value>
inline Value const& ConstantValue<Value>::value() const
{
    return _value;
}

} // namespace fern
