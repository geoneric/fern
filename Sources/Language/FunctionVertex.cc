#include "FunctionVertex.h"



namespace ranally {

FunctionVertex::FunctionVertex(
  UnicodeString const& name,
  ExpressionVertices const& expressions)

  : ExpressionVertex(name),
    _expressions(expressions)

{
}



FunctionVertex::~FunctionVertex()
{
}



ExpressionVertices const& FunctionVertex::expressions() const
{
  return _expressions;
}

} // namespace ranally

