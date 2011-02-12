#include "IfVertex.h"



namespace ranally {
namespace language {

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



boost::shared_ptr<ExpressionVertex> const& IfVertex::condition() const
{
  return _condition;
}



StatementVertices const& IfVertex::trueStatements() const
{
  return _trueStatements;
}



StatementVertices const& IfVertex::falseStatements() const
{
  return _falseStatements;
}

} // namespace language
} // namespace ranally

