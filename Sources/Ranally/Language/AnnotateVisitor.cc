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
  FunctionVertex& /* vertex */)
{
  // Figure out what the properties are of the function. Annotate the vertex
  // with this information.
  // nrArguments
  //   Value types per argument.
  //   Data types per arguments.
  // nrResults
  //   Value types per result.
  //   Data types per result.
  // It is no problem if the operation is not known. Annotation is optional.
  // The validating visitor will check if all information required for
  // execution is present.
  //
  // TODO
  // std::cout << dev::encodeInUTF8(vertex.name()) << std::endl;
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

