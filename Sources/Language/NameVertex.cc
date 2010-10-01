#include "NameVertex.h"



namespace ranally {

NameVertex::NameVertex(
  int lineNr,
  int colId,
  UnicodeString const& name)

  : ExpressionVertex(lineNr, colId),
    _name(name)

{
}



NameVertex::~NameVertex()
{
}

} // namespace ranally

