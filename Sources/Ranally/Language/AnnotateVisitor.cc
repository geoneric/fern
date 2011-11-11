#include "Ranally/Language/AnnotateVisitor.h"
#include <boost/foreach.hpp>
#include "Ranally/Language/Vertices.h"



namespace ranally {
namespace language {

AnnotateVisitor::AnnotateVisitor()

  : Visitor()

{
}



AnnotateVisitor::~AnnotateVisitor()
{
}



void AnnotateVisitor::Visit(
  AssignmentVertex& /* vertex */)
{
}



void AnnotateVisitor::Visit(
  FunctionVertex& /* vertex */)
{
}



void AnnotateVisitor::Visit(
  IfVertex& /* vertex */)
{
}



void AnnotateVisitor::Visit(
  NameVertex& /* vertex */)
{
}



void AnnotateVisitor::Visit(
  OperatorVertex& /* vertex */)
{
}



void AnnotateVisitor::Visit(
  ScriptVertex& vertex)
{
  visitStatements(vertex.statements());
}



void AnnotateVisitor::Visit(
  WhileVertex& /* vertex */)
{
}

} // namespace language
} // namespace ranally

