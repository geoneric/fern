#include "ranally/language/annotate_visitor.h"
#include "ranally/core/string.h"
#include "ranally/operation/result.h"
#include "ranally/language/vertices.h"


namespace ranally {

AnnotateVisitor::AnnotateVisitor(
    ranally::OperationsPtr const& operations)

    : Visitor(),
      _operations(operations)

{
    assert(_operations);
}


void AnnotateVisitor::Visit(
    AssignmentVertex& vertex)
{
    Visitor::Visit(vertex);

    // Proppagate the result types from the expression to the targets.
    ExpressionVertex const& expression(*vertex.expression());
    ExpressionVertex& target(*vertex.target());
    assert(target.resultTypes().empty());
    target.setResultTypes(expression.resultTypes());
}


#define VISIT_NUMBER_VERTEX(                                                   \
    type,                                                                      \
    data_type,                                                                 \
    value_type)                                                                \
void AnnotateVisitor::Visit(                                                   \
    NumberVertex<type>& vertex)                                                \
{                                                                              \
    vertex.addResultType(data_type, value_type);                               \
}

VISIT_NUMBER_VERTEX(int8_t  , DataType::DT_VALUE, VT_INT8   )
VISIT_NUMBER_VERTEX(int16_t , DataType::DT_VALUE, VT_INT16  )
VISIT_NUMBER_VERTEX(int32_t , DataType::DT_VALUE, VT_INT32  )
VISIT_NUMBER_VERTEX(int64_t , DataType::DT_VALUE, VT_INT64  )
VISIT_NUMBER_VERTEX(uint8_t , DataType::DT_VALUE, VT_UINT8  )
VISIT_NUMBER_VERTEX(uint16_t, DataType::DT_VALUE, VT_UINT16 )
VISIT_NUMBER_VERTEX(uint32_t, DataType::DT_VALUE, VT_UINT32 )
VISIT_NUMBER_VERTEX(uint64_t, DataType::DT_VALUE, VT_UINT64 )
VISIT_NUMBER_VERTEX(float   , DataType::DT_VALUE, VT_FLOAT32)
VISIT_NUMBER_VERTEX(double  , DataType::DT_VALUE, VT_FLOAT64)

#undef VISIT_NUMBER_VERTEX


void AnnotateVisitor::Visit(
    OperationVertex& vertex)
{
    if(_operations->has_operation(vertex.name())) {
        assert(!vertex.operation());
        OperationPtr const& operation(_operations->operation(vertex.name()));
        vertex.setOperation(operation);

        for(auto result: operation->results()) {
            vertex.addResultType(result.data_type(), result.value_type());
        }
    }
}

} // namespace ranally
