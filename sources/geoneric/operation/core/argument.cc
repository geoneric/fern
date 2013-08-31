#include "ranally/operation/core/argument.h"


namespace ranally {

Argument::Argument(
    ArgumentType argument_type)

    : _argument_type(argument_type)

{
}


ArgumentType Argument::argument_type() const
{
    return _argument_type;
}

} // namespace ranally
