#include "AssignmentVertex.h"



namespace ranally {

AssignmentVertex::AssignmentVertex(
  ExpressionVertices const& targets,
  ExpressionVertices const& expressions)

  // First (left most) target contains the start position of the assignment
  // statement.
  : SyntaxVertex(targets[0]->line(), targets[0]->col()),
    _targets(targets),
    _expressions(expressions)

{
}



AssignmentVertex::~AssignmentVertex()
{
}

} // namespace ranally

