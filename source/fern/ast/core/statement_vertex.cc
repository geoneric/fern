#include "fern/ast/core/statement_vertex.h"


namespace fern {

StatementVertex::StatementVertex(
    int line_nr,
    int col_id)

    : AstVertex(line_nr, col_id)

{
}

} // namespace fern
