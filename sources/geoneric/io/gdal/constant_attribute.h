#pragma once

#include <memory>
#include "geoneric/io/gdal/attribute.h"
#include "geoneric/io/gdal/constant_value.h"


namespace geoneric {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
template<
    class Value>
class ConstantAttribute:
    public Attribute
{

public:

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
    class Value>
inline ConstantAttribute<Value>::ConstantAttribute()

    : _values(new ConstantValue<Value>())

{
}


template<
    class Value>
inline ConstantAttribute<Value>::ConstantAttribute(
    Value const& value)

    : _values(new ConstantValue<Value>(value))

{
}


template<
    class Value>
void ConstantAttribute<Value>::set(
    Value const& value)
{
    _values->set(value);
}


template<
    class Value>
ConstantValue<Value> const& ConstantAttribute<Value>::values() const
{
    return *_values;
}

} // namespace geoneric
