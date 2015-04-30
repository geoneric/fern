// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/operation/core/argument.h"


namespace fern {
namespace language {

Argument::Argument(
    ArgumentType argument_type)

    : _argument_type(argument_type)

{
}


ArgumentType Argument::argument_type() const
{
    return _argument_type;
}

} // namespace language
} // namespace fern
