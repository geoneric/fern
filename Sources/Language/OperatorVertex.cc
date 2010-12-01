#include "OperatorVertex.h"



namespace ranally {

OperatorVertex::OperatorVertex(
  UnicodeString const& name,
  ExpressionVertices const& expressions)

  : ExpressionVertex(name),
    _symbol(name),
    _expressions(expressions)

{
}



OperatorVertex::~OperatorVertex()
{
}



UnicodeString const& OperatorVertex::symbol() const
{
  return _symbol;
}



ExpressionVertices const& OperatorVertex::expressions() const
{
  return _expressions;
}

} // namespace ranally

