#include "fern/operation/core/attribute_argument.h"
#include "fern/feature/visitor/attribute_type_visitor.h"


namespace fern {

AttributeArgument::AttributeArgument(
    std::shared_ptr<Attribute> const& attribute)

    : Argument(ArgumentType::AT_ATTRIBUTE),
      _attribute(attribute)

{
    fern::AttributeTypeVisitor visitor;
    attribute->Accept(visitor);
    _data_type = visitor.data_type();
    _value_type = visitor.value_type();
}


std::shared_ptr<Attribute> const& AttributeArgument::attribute() const
{
    return _attribute;
}


DataType AttributeArgument::data_type() const
{
    return _data_type;
}


ValueType AttributeArgument::value_type() const
{
    return _value_type;
}

} // namespace fern
