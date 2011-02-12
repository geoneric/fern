#include "StatementVertex.h"



namespace ranally {
namespace language {

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

} // namespace language
} // namespace ranally

