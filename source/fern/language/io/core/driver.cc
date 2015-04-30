// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/io/core/driver.h"


namespace fern {
namespace language {

Driver::Driver(
    std::string const& name)

    : _name(name)

{
}


std::string const& Driver::name() const
{
    return _name;
}

} // namespace language
} // namespace fern
