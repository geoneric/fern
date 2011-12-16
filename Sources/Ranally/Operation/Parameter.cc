#include "Ranally/Operation/Parameter.h"
#include <cassert>



namespace ranally {
namespace operation {

Parameter::Parameter(
  UnicodeString const& name,
  UnicodeString const& description,
  DataTypes dataTypes,
  ValueTypes valueTypes)

  : _name(name),
    _description(description),
    _dataTypes(dataTypes),
    _valueTypes(valueTypes)

{
  assert(!_name.isEmpty());
  assert(!_description.isEmpty());
  assert(_dataTypes != DT_UNKNOWN);
  assert(_valueTypes != VT_UNKNOWN);
}



Parameter::Parameter(
  Parameter const& other)

  : _name(other._name),
    _description(other._description),
    _dataTypes(other._dataTypes),
    _valueTypes(other._valueTypes)

{
}



Parameter& Parameter::operator=(
  Parameter const& other)
{
  if(&other != this) {
    _name = other._name;
    _description = other._description;
    _dataTypes = other._dataTypes;
    _valueTypes = other._valueTypes;
  }

  return *this;
}



Parameter::~Parameter()
{
}



UnicodeString const& Parameter::name() const
{
  return _name;
}



UnicodeString const& Parameter::description() const
{
  return _description;
}



DataTypes Parameter::dataTypes() const
{
  return _dataTypes;
}



ValueTypes Parameter::valueTypes() const
{
  return _valueTypes;
}

} // namespace operation
} // namespace ranally

