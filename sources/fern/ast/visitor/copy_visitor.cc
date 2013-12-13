#include "fern/ast/visitor/copy_visitor.h"
#include "fern/ast/core/vertices.h"


namespace fern {

std::shared_ptr<ModuleVertex> const& CopyVisitor::module_vertex() const
{
    assert(_module_vertex);
    return _module_vertex;
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
        _statements.emplace_back(visitor._statement_vertex);
    }

    assert(_statements.size() == statements.size());
}


void CopyVisitor::Visit(
    AssignmentVertex& /* vertex */)
{
    assert(false);
}


void CopyVisitor::Visit(
    FunctionCallVertex& /* vertex */)
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
    ModuleVertex& /* vertex */)
{
    assert(false);
    // assert(!_module_vertex);
    // visit_statements(vertex.scope()->statements());
    // _module_vertex.reset(new ModuleVertex(vertex.source_name(), _statements));
}


void CopyVisitor::Visit(
    WhileVertex& /* vertex */)
{
    assert(false);
}

} // namespace fern
