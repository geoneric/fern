#include "OperatorVertex.h"



namespace ranally {

OperatorVertex::OperatorVertex(
  UnicodeString const& name,
  ExpressionVertices const& expressions)

  : ExpressionVertex(),
    _name(name),
    _expressions(expressions)

{
}



OperatorVertex::~OperatorVertex()
{
}



UnicodeString const& OperatorVertex::name() const
{
  return _name;
}



ExpressionVertices const& OperatorVertex::expressions() const
{
  return _expressions;
}

} // namespace ranally

