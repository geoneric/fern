// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/ast/core/subscript_vertex.h"


namespace fern {
namespace language {

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


std::string const& SubscriptVertex::symbol() const
{
    return _symbol;
}


} // namespace language
} // namespace fern
