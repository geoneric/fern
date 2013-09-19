#pragma once

#include <map>
#include "geoneric/io/gdal/values.h"


namespace geoneric {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
template<
    class Value>
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
    class Value>
inline ConstantValue<Value>::ConstantValue(
    Value const& value)

    : Values(),
      _value(value)

{
}


template<
    class Value>
inline void ConstantValue<Value>::set(
    Value const& value)
{
    _value = value;
}


template<
    class Value>
inline Value const& ConstantValue<Value>::value() const
{
    return _value;
}

} // namespace geoneric
