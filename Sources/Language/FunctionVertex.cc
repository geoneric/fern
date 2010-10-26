#include "FunctionVertex.h"



namespace ranally {

FunctionVertex::FunctionVertex(
  UnicodeString const& name,
  ExpressionVertices const& expressions)

  : ExpressionVertex(),
    _name(name),
    _expressions(expressions)

{
}



FunctionVertex::~FunctionVertex()
{
}



UnicodeString const& FunctionVertex::name() const
{
  return _name;
}



ExpressionVertices const& FunctionVertex::expressions() const
{
  return _expressions;
}

} // namespace ranally

