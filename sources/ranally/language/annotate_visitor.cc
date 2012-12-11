#include "ranally/language/annotate_visitor.h"
#include "ranally/core/string.h"
#include "ranally/operation/result.h"
#include "ranally/language/vertices.h"


namespace ranally {

AnnotateVisitor::AnnotateVisitor(
    ranally::OperationsPtr const& operations)

    : Visitor(),
      _mode(Mode::Using),
      _result_types_changed(true),
      _operations(operations)

{
    assert(_operations);
}


void AnnotateVisitor::Visit(
    AssignmentVertex& vertex)
{
    assert(_mode == Mode::Using);
    vertex.expression()->Accept(*this);

    _mode = Mode::Defining;
    // Propagate the result types from the expression to the target.
    ExpressionVertex const& expression(*vertex.expression());
    ExpressionVertex& target(*vertex.target());
    if(target.result_types() != expression.result_types()) {
        target.set_result_types(expression.result_types());
        _result_types_changed = true;
        vertex.target()->Accept(*this);
    }

    _mode = Mode::Using;
}


#define VISIT_NUMBER_VERTEX(                                                   \
    type,                                                                      \
    data_type,                                                                 \
    value_type)                                                                \
void AnnotateVisitor::Visit(                                                   \
    NumberVertex<type>& vertex)                                                \
{                                                                              \
    switch(_mode) {                                                            \
        case Using: {                                                          \
            if(vertex.result_types().empty()) {                                \
                vertex.add_result_type(data_type, value_type);                 \
                _result_types_changed = true;                                  \
            }                                                                  \
            break;                                                             \
        }                                                                      \
        case Defining: {                                                       \
            break;                                                             \
        }                                                                      \
    }                                                                          \
}

VISIT_NUMBER_VERTEX(int8_t  , DataTypes::SCALAR, ValueTypes::INT8   )
VISIT_NUMBER_VERTEX(int16_t , DataTypes::SCALAR, ValueTypes::INT16  )
VISIT_NUMBER_VERTEX(int32_t , DataTypes::SCALAR, ValueTypes::INT32  )
VISIT_NUMBER_VERTEX(int64_t , DataTypes::SCALAR, ValueTypes::INT64  )
VISIT_NUMBER_VERTEX(uint8_t , DataTypes::SCALAR, ValueTypes::UINT8  )
VISIT_NUMBER_VERTEX(uint16_t, DataTypes::SCALAR, ValueTypes::UINT16 )
VISIT_NUMBER_VERTEX(uint32_t, DataTypes::SCALAR, ValueTypes::UINT32 )
VISIT_NUMBER_VERTEX(uint64_t, DataTypes::SCALAR, ValueTypes::UINT64 )
VISIT_NUMBER_VERTEX(float   , DataTypes::SCALAR, ValueTypes::FLOAT32)
VISIT_NUMBER_VERTEX(double  , DataTypes::SCALAR, ValueTypes::FLOAT64)

#undef VISIT_NUMBER_VERTEX


void AnnotateVisitor::Visit(
    OperationVertex& vertex)
{
    switch(_mode) {
        case Using: {
            // Depth first, visit the children first.
            Visitor::Visit(vertex);

            // Retrieve info about the operation, if available.
            if(_operations->has_operation(vertex.name())) {
                if(!vertex.operation()) {
                    OperationPtr const& operation(_operations->operation(
                        vertex.name()));
                    vertex.set_operation(operation);

                    assert(vertex.result_types().empty());
                    for(auto result: operation->results()) {
                        vertex.add_result_type(result.data_type(),
                            result.value_type());
                    }

                    _result_types_changed = true;
                }

                // TODO Calculate result type based on result type calculation
                //      strategy (property of the operation) and the result
                //      types of the arguments.
                //      Only update operation's result type if this is
                //      different from the current result type. Set
                //      _result_types_changed accordingly.
            }
            break;
        }
        case Defining: {
            break;
        }
    }
}


void AnnotateVisitor::Visit(
    NameVertex& vertex)
{
    switch(_mode) {
        case Using: {
            break;
        }
        case Defining: {
            for(NameVertex* use_vertex: vertex.uses()) {
                // Propagate the result types to all use sites, but only when
                // needed.
                assert(use_vertex);
                if(use_vertex->result_types() != vertex.result_types()) {
                    use_vertex->set_result_types(vertex.result_types());
                    _result_types_changed = true;
                }
            }
            break;
        }
    }
}


void AnnotateVisitor::Visit(
    SubscriptVertex& vertex)
{
    switch(_mode) {
        case Using: {
            // The result type of a subscript expression is the same as the
            // result type of the main expression.
            if(vertex.result_types() != vertex.expression()->result_types()) {
                vertex.set_result_types(vertex.expression()->result_types());
                _result_types_changed = true;
            }
            break;
        }
        case Defining: {
            break;
        }
    }
}


void AnnotateVisitor::Visit(
    ScriptVertex& vertex)
{
    _mode = Mode::Using;
    _result_types_changed = true;

    while(_result_types_changed) {
        _result_types_changed = false;
        Visitor::Visit(vertex);
    }
}

} // namespace ranally
