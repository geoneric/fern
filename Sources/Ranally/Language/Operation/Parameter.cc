#include "Ranally/Language/Operation/Parameter.h"


namespace ranally {

Parameter::Parameter()
{
}


Parameter::~Parameter()
{
}


ValueTypes Parameter::valueTypes() const
{
    return _valueTypes;
}


DataTypes Parameter::dataTypes() const
{
    return _dataTypes;
}

} // namespace ranally
