#include "Ranally/Language/IfVertex.h"


namespace ranally {

IfVertex::IfVertex(
    std::shared_ptr<ExpressionVertex> const& condition,
    StatementVertices const& trueStatements,
    StatementVertices const& falseStatements)

    : StatementVertex(),
      _condition(condition),
      _trueStatements(trueStatements),
      _falseStatements(falseStatements)

{
}


std::shared_ptr<ExpressionVertex> const& IfVertex::condition() const
{
    return _condition;
}


StatementVertices const& IfVertex::trueStatements() const
{
    return _trueStatements;
}


StatementVertices& IfVertex::trueStatements()
{
    return _trueStatements;
}


StatementVertices const& IfVertex::falseStatements() const
{
    return _falseStatements;
}


StatementVertices& IfVertex::falseStatements()
{
    return _falseStatements;
}

} // namespace ranally
