#include "geoneric/feature/attribute.h"
#include <cassert>


namespace geoneric {

Attribute::Attribute(
    DataType data_type,
    ValueType value_type)

    : _data_type(data_type),
      _value_type(value_type)

{
}


DataType Attribute::data_type() const
{
    return _data_type;
}


ValueType Attribute::value_type() const
{
    return _value_type;
}

} // namespace geoneric
