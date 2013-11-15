#include "fern/operation/core/argument.h"


namespace fern {

Argument::Argument(
    ArgumentType argument_type)

    : _argument_type(argument_type)

{
}


ArgumentType Argument::argument_type() const
{
    return _argument_type;
}

} // namespace fern
