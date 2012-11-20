#include "Ranally/Operation/result.h"
#include <cassert>


namespace ranally {

Result::Result(
    String const& name,
    String const& description,
    DataType const& dataType,
    ValueType const& valueType)

    : _name(name),
      _description(description),
      _dataType(dataType),
      _valueType(valueType)

{
    assert(!_name.isEmpty());
    assert(!_description.isEmpty());
}


Result::Result(
    Result const& other)

    : _name(other._name),
      _description(other._description),
      _dataType(other._dataType),
      _valueType(other._valueType)

{
}


Result& Result::operator=(
  Result const& other)
{
    if(&other != this) {
        _name = other._name;
        _description = other._description;
        _dataType = other._dataType;
        _valueType = other._valueType;
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


DataType Result::dataType() const
{
    return _dataType;
}


ValueType Result::valueType() const
{
    return _valueType;
}

} // namespace ranally
