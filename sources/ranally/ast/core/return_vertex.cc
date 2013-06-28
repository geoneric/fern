#include "ranally/ast/core/return_vertex.h"


namespace ranally {

ReturnVertex::ReturnVertex(
    ExpressionVertexPtr const& expression)

    : StatementVertex(),
      _expression(expression)

{
}


ExpressionVertexPtr const& ReturnVertex::expression() const
{
    return _expression;
}

} // namespace ranally
