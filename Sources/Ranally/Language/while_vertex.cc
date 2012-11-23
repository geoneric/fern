#include "Ranally/Language/while_vertex.h"


namespace ranally {

WhileVertex::WhileVertex(
    std::shared_ptr<ExpressionVertex> const& condition,
    StatementVertices const& trueStatements,
    StatementVertices const& falseStatements)

    : StatementVertex(),
      _condition(condition),
      _trueStatements(trueStatements),
      _falseStatements(falseStatements)

{
}


std::shared_ptr<ExpressionVertex> const& WhileVertex::condition() const
{
    return _condition;
}


StatementVertices const& WhileVertex::trueStatements() const
{
    return _trueStatements;
}


StatementVertices& WhileVertex::trueStatements()
{
    return _trueStatements;
}


StatementVertices const& WhileVertex::falseStatements() const
{
    return _falseStatements;
}


StatementVertices& WhileVertex::falseStatements()
{
    return _falseStatements;
}

} // namespace ranally
