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



UnicodeString const& NameVertex::name() const
{
  return _name;
}

} // namespace ranally

