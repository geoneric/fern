#include "geoneric/ast/core/return_vertex.h"


namespace geoneric {

ReturnVertex::ReturnVertex()

    : StatementVertex(),
      _expression()

{
}


ReturnVertex::ReturnVertex(
    ExpressionVertexPtr const& expression)

    : StatementVertex(),
      _expression(expression)

{
}


//! Return the expression returned, which may be absent.
/*!
  \return    Expression if set, or null pointer.
*/
ExpressionVertexPtr const& ReturnVertex::expression() const
{
    return _expression;
}

} // namespace geoneric
