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
  AssignmentVertex& vertex)
{
  Visitor::Visit(vertex);

  // Proppagate the result types from the expression to the targets.
  ExpressionVertices const& expressions(vertex.expressions());
  ExpressionVertices& targets(vertex.targets());
  assert(expressions.size() == targets.size());

  for(size_t i = 0; i < expressions.size(); ++i) {
    ExpressionVertexPtr const& expression(expressions[i]);
    ExpressionVertexPtr& target(targets[i]);
    assert(target->resultTypes().empty());
    target->setResultTypes(expression->resultTypes());
  }
}



#define VISIT_NUMBER_VERTEX(                                                   \
  type,                                                                        \
  dataType,                                                                    \
  valueType)                                                                   \
void AnnotateVisitor::Visit(                                                   \
  NumberVertex<type>& vertex)                                                  \
{                                                                              \
  vertex.addResultType(operation::dataType, operation::valueType);             \
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
      vertex.addResultType(result.dataType(), result.valueType());
    }
  }
}

} // namespace language
} // namespace ranally

