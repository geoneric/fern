// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/ast/core/string_vertex.h"


namespace fern {
namespace language {

StringVertex::StringVertex(
    int line_nr,
    int col_id,
    std::string const& value)

    : ExpressionVertex(line_nr, col_id, "\"" + value + "\""),
      _value(value)

{
}


std::string const& StringVertex::value() const
{
    return _value;
}

} // namespace language
} // namespace fern
