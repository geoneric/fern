#include "IfVertex.h"



namespace ranally {

IfVertex::IfVertex(
  boost::shared_ptr<ExpressionVertex> const& condition,
  StatementVertices const& trueStatements,
  StatementVertices const& falseStatements)

  : StatementVertex(),
    _condition(condition),
    _trueStatements(trueStatements),
    _falseStatements(falseStatements)

{
}



IfVertex::~IfVertex()
{
}

} // namespace ranally

