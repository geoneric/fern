// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/ast/core/statement_vertex.h"


namespace fern {

StatementVertex::StatementVertex(
    int line_nr,
    int col_id)

    : AstVertex(line_nr, col_id)

{
}

} // namespace fern
