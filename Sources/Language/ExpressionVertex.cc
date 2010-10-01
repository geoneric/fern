#include "ExpressionVertex.h"



namespace ranally {

ExpressionVertex::ExpressionVertex(
  int lineNr,
  int colId)

  : SyntaxVertex(lineNr, colId)

{
}



ExpressionVertex::~ExpressionVertex()
{
}

} // namespace ranally

