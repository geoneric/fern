#include "ranally/operation/parameter.h"
#include <cassert>


namespace ranally {

Parameter::Parameter(
    String const& name,
    String const& description,
    DataTypes data_types,
    ValueTypes value_types)

    : _name(name),
      _description(description),
      _data_types(data_types),
      _value_types(value_types)

{
    assert(!_name.is_empty());
    assert(!_description.is_empty());
    assert(_data_types != DataType::DT_UNKNOWN);
    assert(_value_types != VT_UNKNOWN);
}


Parameter::Parameter(
    Parameter const& other)

    : _name(other._name),
      _description(other._description),
      _data_types(other._data_types),
      _value_types(other._value_types)

{
}


Parameter& Parameter::operator=(
    Parameter const& other)
{
    if(&other != this) {
        _name = other._name;
        _description = other._description;
        _data_types = other._data_types;
        _value_types = other._value_types;
    }

    return *this;
}


Parameter::~Parameter()
{
}


String const& Parameter::name() const
{
    return _name;
}


String const& Parameter::description() const
{
    return _description;
}


DataTypes Parameter::data_types() const
{
    return _data_types;
}


ValueTypes Parameter::value_types() const
{
    return _value_types;
}

} // namespace ranally
