#include "WhileVertex.h"



namespace ranally {

WhileVertex::WhileVertex(
  boost::shared_ptr<ExpressionVertex> const& condition,
  StatementVertices const& trueStatements,
  StatementVertices const& falseStatements)

  : StatementVertex(),
    _condition(condition),
    _trueStatements(trueStatements),
    _falseStatements(falseStatements)

{
}



WhileVertex::~WhileVertex()
{
}



boost::shared_ptr<ranally::ExpressionVertex> const& WhileVertex::condition() const
{
  return _condition;
}



StatementVertices const& WhileVertex::trueStatements() const
{
  return _trueStatements;
}



StatementVertices const& WhileVertex::falseStatements() const
{
  return _falseStatements;
}

} // namespace ranally

