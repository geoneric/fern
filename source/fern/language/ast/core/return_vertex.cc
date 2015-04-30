// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/ast/core/return_vertex.h"


namespace fern {
namespace language {

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

} // namespace language
} // namespace fern
