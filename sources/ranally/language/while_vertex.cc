#include "ranally/language/while_vertex.h"


namespace ranally {

WhileVertex::WhileVertex(
    std::shared_ptr<ExpressionVertex> const& condition,
    StatementVertices const& true_statements,
    StatementVertices const& false_statements)

    : StatementVertex(),
      _condition(condition),
      _true_statements(true_statements),
      _false_statements(false_statements)

{
}


std::shared_ptr<ExpressionVertex> const& WhileVertex::condition() const
{
    return _condition;
}


StatementVertices const& WhileVertex::true_statements() const
{
    return _true_statements;
}


StatementVertices& WhileVertex::true_statements()
{
    return _true_statements;
}


StatementVertices const& WhileVertex::false_statements() const
{
    return _false_statements;
}


StatementVertices& WhileVertex::false_statements()
{
    return _false_statements;
}

} // namespace ranally
