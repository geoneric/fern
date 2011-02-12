#include "Ranally/Language/ExpressionVertex.h"



namespace ranally {
namespace language {

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

} // namespace language
} // namespace ranally

