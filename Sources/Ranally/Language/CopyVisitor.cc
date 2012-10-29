#include "Ranally/Language/CopyVisitor.h"
#include <boost/foreach.hpp>
#include "Ranally/Language/Vertices.h"


namespace ranally {
namespace language {

CopyVisitor::CopyVisitor()

    : Visitor()

{
}


CopyVisitor::~CopyVisitor()
{
}


boost::shared_ptr<ScriptVertex> const& CopyVisitor::scriptVertex() const
{
    assert(_scriptVertex);
    return _scriptVertex;
}


// SyntaxVertices const& CopyVisitor::syntaxVertices() const
// {
//   return _syntaxVertices;
// }


// StatementVertices const& CopyVisitor::statements() const
// {
//   return _statements;
// }


void CopyVisitor::visitStatements(
    StatementVertices& statements)
{
    assert(_statements.empty());

    BOOST_FOREACH(boost::shared_ptr<StatementVertex> const& statement,
            statements) {
        CopyVisitor visitor;
        statement->Accept(visitor);
        assert(_statementVertex);
        _statements.push_back(visitor._statementVertex);
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
    ScriptVertex& vertex)
{
    assert(!_scriptVertex);
    visitStatements(vertex.statements());
    _scriptVertex.reset(new ScriptVertex(vertex.sourceName(), _statements));
}


void CopyVisitor::Visit(
    WhileVertex& /* vertex */)
{
    assert(false);
}

} // namespace language
} // namespace ranally
