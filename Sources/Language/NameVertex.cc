#include "NameVertex.h"



namespace ranally {

NameVertex::NameVertex(
  int lineNr,
  int colId,
  UnicodeString const& name)

  : ExpressionVertex(lineNr, colId, name)

{
}



NameVertex::~NameVertex()
{
}

} // namespace ranally

