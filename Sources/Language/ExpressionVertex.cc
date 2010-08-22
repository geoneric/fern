#include "ExpressionVertex.h"



namespace ranally {

ExpressionVertex::ExpressionVertex(
  int lineNr,
  int colId,
  UnicodeString const& value)

  : SyntaxVertex(lineNr, colId, value)

{
}



ExpressionVertex::~ExpressionVertex()
{
}

} // namespace ranally

