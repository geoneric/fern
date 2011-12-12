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

} // namespace operation
} // namespace ranally

