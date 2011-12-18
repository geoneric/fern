#include "Ranally/Language/AnnotateVisitor.h"
#include <boost/foreach.hpp>
#include "dev_UnicodeUtils.h"
#include "Ranally/Language/Vertices.h"



namespace ranally {
namespace language {

AnnotateVisitor::AnnotateVisitor(
  ranally::operation::OperationsPtr const& operations)

  : Visitor(),
    _operations(operations)

{
  assert(_operations);
}



AnnotateVisitor::~AnnotateVisitor()
{
}



void AnnotateVisitor::Visit(
  AssignmentVertex& /* vertex */)
{
}



void AnnotateVisitor::Visit(
  FunctionVertex& vertex)
{
  if(_operations->hasOperation(vertex.name())) {
    assert(!vertex.operation());
    vertex.setOperation(_operations->operation(vertex.name()));
  }
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

