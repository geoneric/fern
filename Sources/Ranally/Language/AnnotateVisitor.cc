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
  NumberVertex<int8_t>& vertex)
{
  vertex.setDataType(operation::DT_VALUE);
  vertex.setValueType(operation::VT_INT8);
}



void AnnotateVisitor::Visit(
  NumberVertex<int16_t>& vertex)
{
  vertex.setDataType(operation::DT_VALUE);
  vertex.setValueType(operation::VT_INT16);
}



void AnnotateVisitor::Visit(
  NumberVertex<int32_t>& vertex)
{
  vertex.setDataType(operation::DT_VALUE);
  vertex.setValueType(operation::VT_INT32);
}



void AnnotateVisitor::Visit(
  NumberVertex<int64_t>& vertex)
{
  vertex.setDataType(operation::DT_VALUE);
  vertex.setValueType(operation::VT_INT64);
}



void AnnotateVisitor::Visit(
  NumberVertex<uint8_t>& vertex)
{
  vertex.setDataType(operation::DT_VALUE);
  vertex.setValueType(operation::VT_UINT8);
}



void AnnotateVisitor::Visit(
  NumberVertex<uint16_t>& vertex)
{
  vertex.setDataType(operation::DT_VALUE);
  vertex.setValueType(operation::VT_UINT16);
}



void AnnotateVisitor::Visit(
  NumberVertex<uint32_t>& vertex)
{
  vertex.setDataType(operation::DT_VALUE);
  vertex.setValueType(operation::VT_UINT32);
}



void AnnotateVisitor::Visit(
  NumberVertex<uint64_t>& vertex)
{
  vertex.setDataType(operation::DT_VALUE);
  vertex.setValueType(operation::VT_UINT64);
}



void AnnotateVisitor::Visit(
  NumberVertex<float>& vertex)
{
  vertex.setDataType(operation::DT_VALUE);
  vertex.setValueType(operation::VT_FLOAT32);
}



void AnnotateVisitor::Visit(
  NumberVertex<double>& vertex)
{
  vertex.setDataType(operation::DT_VALUE);
  vertex.setValueType(operation::VT_FLOAT64);
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

