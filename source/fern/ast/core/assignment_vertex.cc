// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/ast/core/assignment_vertex.h"


namespace fern {

AssignmentVertex::AssignmentVertex(
    ExpressionVertexPtr const& target,
    ExpressionVertexPtr const& expression)

    // First (left most) target contains the start position of the assignment
    // statement.
    : StatementVertex(target->line(), target->col()),
      _target(target),
      _expression(expression)

{
    assert(_target);
    assert(_expression);
    _target->set_value(_expression);
}


ExpressionVertexPtr const& AssignmentVertex::target() const
{
    return _target;
}


ExpressionVertexPtr& AssignmentVertex::target()
{
    return _target;
}


void AssignmentVertex::set_expression(
    ExpressionVertexPtr const& expression)
{
    assert(expression);
    _expression = expression;
}


ExpressionVertexPtr const& AssignmentVertex::expression() const
{
    return _expression;
}

} // namespace fern
