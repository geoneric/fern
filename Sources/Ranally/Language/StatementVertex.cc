#include "Ranally/Language/StatementVertex.h"


namespace ranally {

StatementVertex::StatementVertex()

    : SyntaxVertex()

{
}


StatementVertex::StatementVertex(
    int lineNr,
    int colId)

    : SyntaxVertex(lineNr, colId)

{
}


StatementVertex::~StatementVertex()
{
}

} // namespace ranally
