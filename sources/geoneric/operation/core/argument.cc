#include "geoneric/operation/core/argument.h"


namespace geoneric {

Argument::Argument(
    ArgumentType argument_type)

    : _argument_type(argument_type)

{
}


ArgumentType Argument::argument_type() const
{
    return _argument_type;
}

} // namespace geoneric
