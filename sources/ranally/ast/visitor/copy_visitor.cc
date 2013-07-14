#include "ranally/ast/visitor/copy_visitor.h"
#include "ranally/ast/core/vertices.h"


namespace ranally {

std::shared_ptr<ScriptVertex> const& CopyVisitor::script_vertex() const
{
    assert(_script_vertex);
    return _script_vertex;
}


// SyntaxVertices const& CopyVisitor::syntaxVertices() const
// {
//   return _syntaxVertices;
// }


// StatementVertices const& CopyVisitor::statements() const
// {
//   return _statements;
// }


void CopyVisitor::visit_statements(
    StatementVertices& statements)
{
    assert(_statements.empty());

    for(auto statement: statements) {
        CopyVisitor visitor;
        statement->Accept(visitor);
        assert(_statement_vertex);
        _statements.push_back(visitor._statement_vertex);
    }

    assert(_statements.size() == statements.size());
}


void CopyVisitor::Visit(
    AssignmentVertex& /* vertex */)
{
    assert(false);
}


void CopyVisitor::Visit(
    FunctionVertex& /* vertex */)
{
    assert(false);
}


void CopyVisitor::Visit(
    IfVertex& /* vertex */)
{
    assert(false);
}


void CopyVisitor::Visit(
    NameVertex& /* vertex */)
{
    assert(false);
}


void CopyVisitor::Visit(
    OperatorVertex& /* vertex */)
{
    assert(false);
}


void CopyVisitor::Visit(
    ScriptVertex& /* vertex */)
{
    assert(false);
    // assert(!_script_vertex);
    // visit_statements(vertex.scope()->statements());
    // _script_vertex.reset(new ScriptVertex(vertex.source_name(), _statements));
}


void CopyVisitor::Visit(
    WhileVertex& /* vertex */)
{
    assert(false);
}

} // namespace ranally
