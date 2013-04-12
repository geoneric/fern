#include "ranally/ast/core/if_vertex.h"


namespace ranally {

IfVertex::IfVertex(
    std::shared_ptr<ExpressionVertex> const& condition,
    StatementVertices const& true_statements,
    StatementVertices const& false_statements)

    : StatementVertex(),
      _condition(condition),
      _true_statements(true_statements),
      _false_statements(false_statements)

{
}


std::shared_ptr<ExpressionVertex> const& IfVertex::condition() const
{
    return _condition;
}


StatementVertices const& IfVertex::true_statements() const
{
    return _true_statements;
}


StatementVertices& IfVertex::true_statements()
{
    return _true_statements;
}


StatementVertices const& IfVertex::false_statements() const
{
    return _false_statements;
}


StatementVertices& IfVertex::false_statements()
{
    return _false_statements;
}

} // namespace ranally
