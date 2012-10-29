#include "Ranally/Language/AssignmentVertex.h"


namespace ranally {
namespace language {

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
    _target->setValue(_expression);
}


AssignmentVertex::~AssignmentVertex()
{
}


ExpressionVertexPtr const& AssignmentVertex::target() const
{
    return _target;
}


ExpressionVertexPtr& AssignmentVertex::target()
{
    return _target;
}


void AssignmentVertex::setExpression(
    ExpressionVertexPtr const& expression)
{
    assert(expression);
    _expression = expression;
}


ExpressionVertexPtr const& AssignmentVertex::expression() const
{
    return _expression;
}

} // namespace language
} // namespace ranally
