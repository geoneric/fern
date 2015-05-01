// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <iostream>


namespace fern {
namespace language {

enum class ArgumentType {

    AT_ATTRIBUTE,

    AT_FEATURE

};


std::ostream&      operator<<          (std::ostream& stream,
                                        ArgumentType const& argument_type);

} // namespace language
} // namespace fern
