#include "ranally/language/result_type.h"


namespace ranally {

ResultType::ResultType(
    DataTypes const& data_types,
    ValueTypes const& value_types)

    : _data_types(data_types),
      _value_types(value_types)

{
}


DataTypes ResultType::data_type() const
{
    return _data_types;
}


ValueTypes ResultType::value_type() const
{
    return _value_types;
}


bool ResultType::fixed() const
{
    return _data_types.fixed() && _value_types.fixed();
}

} // namespace ranally
