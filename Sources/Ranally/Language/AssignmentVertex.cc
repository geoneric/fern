#include "Ranally/Language/AssignmentVertex.h"



namespace ranally {
namespace language {

AssignmentVertex::AssignmentVertex(
  ExpressionVertices const& targets,
  ExpressionVertices const& expressions)

  // First (left most) target contains the start position of the assignment
  // statement.
  : StatementVertex(targets[0]->line(), targets[0]->col()),
    _targets(targets),
    _expressions(expressions)

{
}



AssignmentVertex::~AssignmentVertex()
{
}



ExpressionVertices const& AssignmentVertex::targets() const
{
  return _targets;
}



ExpressionVertices const& AssignmentVertex::expressions() const
{
  return _expressions;
}

} // namespace language
} // namespace ranally
