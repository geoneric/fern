// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/ast/core/attribute_vertex.h"


namespace fern {

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

} // namespace fern
