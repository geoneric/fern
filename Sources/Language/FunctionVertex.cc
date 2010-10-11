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

} // namespace ranally

