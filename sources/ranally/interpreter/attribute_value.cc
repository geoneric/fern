#include "ranally/interpreter/attribute_value.h"


namespace ranally {
namespace interpreter {

AttributeValue::AttributeValue(
    std::shared_ptr<Attribute> const& attribute)

    : Value(ValueType::VT_ATTRIBUTE),
      _attribute(attribute)

{
}


std::shared_ptr<Attribute> const& AttributeValue::attribute() const
{
    return _attribute;
}

} // namespace interpreter
} // namespace ranally
