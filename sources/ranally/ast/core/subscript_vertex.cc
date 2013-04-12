#include "ranally/ast/core/subscript_vertex.h"


namespace ranally {

SubscriptVertex::SubscriptVertex(
    ExpressionVertexPtr const& expression,
    ExpressionVertexPtr const& selection)

    : ExpressionVertex("Subscript"),
      _symbol("[]"),
      _expression(expression),
      _selection(selection)

{
}


ExpressionVertexPtr const& SubscriptVertex::expression() const
{
    return _expression;
}


ExpressionVertexPtr const& SubscriptVertex::selection() const
{
    return _selection;
}


String const& SubscriptVertex::symbol() const
{
    return _symbol;
}


} // namespace ranally
