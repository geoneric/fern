#include "geoneric/ast/core/attribute_vertex.h"


namespace geoneric {

AttributeVertex::AttributeVertex(
    ExpressionVertexPtr const& expression,
    String const& member_name)

    : ExpressionVertex("Attribute"),
      _symbol("."),
      _expression(expression),
      _member_name(member_name)

{
}


ExpressionVertexPtr const& AttributeVertex::expression() const
{
    return _expression;
}


String const& AttributeVertex::member_name() const
{
    return _member_name;
}


String const& AttributeVertex::symbol() const
{
    return _symbol;
}

} // namespace geoneric
