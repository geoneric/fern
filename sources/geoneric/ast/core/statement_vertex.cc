#include "ranally/ast/core/statement_vertex.h"


namespace ranally {

StatementVertex::StatementVertex(
    int line_nr,
    int col_id)

    : AstVertex(line_nr, col_id)

{
}

} // namespace ranally
