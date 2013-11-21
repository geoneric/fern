#include "fern/compiler/argument.h"


namespace fern {

Argument::Argument(
    String const& name)

    : _name(name)

{
}


String const& Argument::name() const
{
    return _name;
}

} // namespace fern
