#include "ranally/operation/result.h"
#include <cassert>


namespace ranally {

Result::Result(
    String const& name,
    String const& description,
    DataTypes const& data_type,
    ValueTypes const& value_type)

    : _name(name),
      _description(description),
      _data_type(data_type),
      _value_type(value_type)

{
    assert(!_name.is_empty());
    assert(!_description.is_empty());
}


Result::Result(
    Result const& other)

    : _name(other._name),
      _description(other._description),
      _data_type(other._data_type),
      _value_type(other._value_type)

{
}


Result& Result::operator=(
  Result const& other)
{
    if(&other != this) {
        _name = other._name;
        _description = other._description;
        _data_type = other._data_type;
        _value_type = other._value_type;
    }

    return *this;
}


String const& Result::name() const
{
    return _name;
}


String const& Result::description() const
{
    return _description;
}


DataTypes Result::data_type() const
{
    return _data_type;
}


ValueTypes Result::value_type() const
{
    return _value_type;
}

} // namespace ranally
