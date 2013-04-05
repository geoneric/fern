#include "ranally/interpreter/value.h"


namespace ranally {
namespace interpreter {

Value::Value(
    ValueType value_type)

    : _value_type(value_type)

{
}


ValueType Value::value_type() const
{
    return _value_type;
}

} // namespace interpreter
} // namespace ranally
