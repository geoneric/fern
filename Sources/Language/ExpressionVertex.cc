#include "ExpressionVertex.h"



namespace ranally {

ExpressionVertex::ExpressionVertex(
  UnicodeString const& name)

  : StatementVertex(),
    _name(name)

{
}



ExpressionVertex::ExpressionVertex(
  int lineNr,
  int colId,
  UnicodeString const& name)

  : StatementVertex(lineNr, colId),
    _name(name)

{
}



ExpressionVertex::~ExpressionVertex()
{
}



UnicodeString const& ExpressionVertex::name() const
{
  return _name;
}

} // namespace ranally

