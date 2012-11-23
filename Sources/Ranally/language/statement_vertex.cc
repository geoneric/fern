#include "ranally/language/statement_vertex.h"


namespace ranally {

StatementVertex::StatementVertex(
    int lineNr,
    int colId)

    : SyntaxVertex(lineNr, colId)

{
}

} // namespace ranally
