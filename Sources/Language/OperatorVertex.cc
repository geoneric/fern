#include "OperatorVertex.h"



namespace ranally {

OperatorVertex::OperatorVertex(
  UnicodeString const& name,
  ExpressionVertices const& expressions)

  : ExpressionVertex(name),
    _expressions(expressions)

{
}



OperatorVertex::~OperatorVertex()
{
}



ExpressionVertices const& OperatorVertex::expressions() const
{
  return _expressions;
}

} // namespace ranally

