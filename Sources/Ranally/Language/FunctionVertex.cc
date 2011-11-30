#include "Ranally/Language/FunctionVertex.h"



namespace ranally {
namespace language {

FunctionVertex::FunctionVertex(
  UnicodeString const& name,
  ExpressionVertices const& expressions)

  : OperationVertex(name),
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

} // namespace language
} // namespace ranally

