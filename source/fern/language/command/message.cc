// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/command/message.h"
#include <iostream>
#include "fern/configure.h"


namespace fern {
namespace language {

void show_version()
{
    std::cout << "fern " << FERN_VERSION << "\n";
    std::cout << "Copyright " << FERN_COPYRIGHT << "\n";
}

} // namespace language
} // namespace fern
