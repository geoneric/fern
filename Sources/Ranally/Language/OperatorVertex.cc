#include "Ranally/Language/OperatorVertex.h"



namespace ranally {
namespace language {

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

} // namespace language
} // namespace ranally

