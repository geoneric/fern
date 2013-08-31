#include "geoneric/operation/core/attribute_argument.h"


namespace geoneric {

AttributeArgument::AttributeArgument(
    std::shared_ptr<Attribute> const& attribute)

    : Argument(ArgumentType::AT_ATTRIBUTE),
      _attribute(attribute)

{
}


std::shared_ptr<Attribute> const& AttributeArgument::attribute() const
{
    return _attribute;
}

} // namespace geoneric
