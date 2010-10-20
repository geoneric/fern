#include "ExpressionVertex.h"



namespace ranally {

ExpressionVertex::ExpressionVertex()

  : StatementVertex()

{
}



ExpressionVertex::ExpressionVertex(
  int lineNr,
  int colId)

  : StatementVertex(lineNr, colId)

{
}



ExpressionVertex::~ExpressionVertex()
{
}

} // namespace ranally

