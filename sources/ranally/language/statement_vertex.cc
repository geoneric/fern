#include "ranally/language/statement_vertex.h"


namespace ranally {

StatementVertex::StatementVertex(
    int line_nr,
    int col_id)

    : SyntaxVertex(line_nr, col_id)

{
}

} // namespace ranally
