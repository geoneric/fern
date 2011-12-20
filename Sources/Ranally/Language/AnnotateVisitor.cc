#include "Ranally/Language/AnnotateVisitor.h"
#include <boost/foreach.hpp>
#include "dev_UnicodeUtils.h"
#include "Ranally/Operation/Result.h"
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



// void AnnotateVisitor::Visit(
//   FunctionVertex& vertex)
// {
//   if(_operations->hasOperation(vertex.name())) {
//     assert(!vertex.operation());
//     operation::OperationPtr const& operation(
//       _operations->operation(vertex.name()));
//     vertex.setOperation(operation);
// 
//     BOOST_FOREACH(operation::Result const& result, operation->results()) {
//       vertex.addResult(result.dataType(), result.valueType());
//     }
//   }
// }



void AnnotateVisitor::Visit(
  IfVertex& /* vertex */)
{
}



void AnnotateVisitor::Visit(
  NameVertex& /* vertex */)
{
}



#define VISIT_NUMBER_VERTEX(                                                   \
  type,                                                                        \
  dataType,                                                                    \
  valueType)                                                                   \
void AnnotateVisitor::Visit(                                                   \
  NumberVertex<type>& vertex)                                                  \
{                                                                              \
  vertex.addResult(operation::dataType, operation::valueType);                 \
}

VISIT_NUMBER_VERTEX(int8_t  , DT_VALUE, VT_INT8   )
VISIT_NUMBER_VERTEX(int16_t , DT_VALUE, VT_INT16  )
VISIT_NUMBER_VERTEX(int32_t , DT_VALUE, VT_INT32  )
VISIT_NUMBER_VERTEX(int64_t , DT_VALUE, VT_INT64  )
VISIT_NUMBER_VERTEX(uint8_t , DT_VALUE, VT_UINT8  )
VISIT_NUMBER_VERTEX(uint16_t, DT_VALUE, VT_UINT16 )
VISIT_NUMBER_VERTEX(uint32_t, DT_VALUE, VT_UINT32 )
VISIT_NUMBER_VERTEX(uint64_t, DT_VALUE, VT_UINT64 )
VISIT_NUMBER_VERTEX(float   , DT_VALUE, VT_FLOAT32)
VISIT_NUMBER_VERTEX(double  , DT_VALUE, VT_FLOAT64)

#undef VISIT_NUMBER_VERTEX



void AnnotateVisitor::Visit(
  OperationVertex& vertex)
{
  if(_operations->hasOperation(vertex.name())) {
    assert(!vertex.operation());
    operation::OperationPtr const& operation(
      _operations->operation(vertex.name()));
    vertex.setOperation(operation);

    BOOST_FOREACH(operation::Result const& result, operation->results()) {
      vertex.addResult(result.dataType(), result.valueType());
    }
  }
}



// void AnnotateVisitor::Visit(
//   OperatorVertex& vertex)
// {
//   if(_operations->hasOperation(vertex.name())) {
//     assert(!vertex.operation());
//     operation::OperationPtr const& operation(
//       _operations->operation(vertex.name()));
//     vertex.setOperation(operation);
// 
//     BOOST_FOREACH(operation::Result const& result, operation->results()) {
//       vertex.addResult(result.dataType(), result.valueType());
//     }
//   }
// }



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

