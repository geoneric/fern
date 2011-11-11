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



StatementVertices const& CopyVisitor::statements() const
{
  return _statements;
}



void CopyVisitor::Visit(
  AssignmentVertex& /* vertex */)
{
}



void CopyVisitor::Visit(
  FunctionVertex& /* vertex */)
{
}



void CopyVisitor::Visit(
  IfVertex& /* vertex */)
{
}



void CopyVisitor::Visit(
  NameVertex& /* vertex */)
{
}



void CopyVisitor::Visit(
  OperatorVertex& /* vertex */)
{
}



void CopyVisitor::Visit(
  ScriptVertex& vertex)
{
  visitStatements(vertex.statements());
}



void CopyVisitor::Visit(
  WhileVertex& /* vertex */)
{
}

} // namespace language
} // namespace ranally

