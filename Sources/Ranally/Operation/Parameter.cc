#include "Ranally/Operation/Parameter.h"



namespace ranally {
namespace operation {

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



std::vector<DataType> const& Parameter::dataTypes() const
{
  return _dataTypes;
}



std::vector<ValueType> const& Parameter::valueTypes() const
{
  return _valueTypes;
}

} // namespace operation
} // namespace ranally

