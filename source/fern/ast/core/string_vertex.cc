// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/ast/core/string_vertex.h"


namespace fern {

StringVertex::StringVertex(
    int line_nr,
    int col_id,
    String const& value)

    : ExpressionVertex(line_nr, col_id, String("\"") + value + String("\"")),
      _value(value)

{
}


String const& StringVertex::value() const
{
    return _value;
}

} // namespace fern
