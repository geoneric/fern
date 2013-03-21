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
    // Let the source expression execute itself, leaving the result(s) on the
    // stack.
    vertex.expression()->Accept(*this);

    // Assume the target expression is a NameVertex (it should, for now).
    assert(dynamic_cast<NameVertex const*>(vertex.target().get()));

    // Store the result in a scoped symbol table for later reference.
    // Update scope at correct moments in other visit functions.
    assert(!_stack.empty());
    _symbol_table.add_value(vertex.target()->name(), _stack.top());
    _stack.pop();

    // // Propagate the result types from the expression to the target.
    // ExpressionVertex const& expression(*vertex.expression());
    // ExpressionVertex& target(*vertex.target());
    // assert(target.result_types().empty());
    // target.set_result_types(expression.result_types());
    // vertex.target()->Accept(*this);
}


#define VISIT_NUMBER_VERTEX(                                                   \
    type,                                                                      \
    data_type,                                                                 \
    value_type)                                                                \
void AnnotateVisitor::Visit(                                                   \
    NumberVertex<type>& vertex)                                                \
{                                                                              \
    assert(vertex.result_types().empty());                                     \
    ResultType result_type(data_type, value_type);                             \
    _stack.push(result_type);                                                  \
    vertex.add_result_type(result_type);                                       \
}

// TODO Use traits in the implementation! Don't pass these things to the macro.
// TODO Refactor all these macro call lists. They are everywhere. Make a
//      VISIT_NUMBER_VERTICES macro that calls the macro for each numeric
//      type.
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
    // Depth first, visit the children first.
    Visitor::Visit(vertex);

    // Retrieve info about the operation, if available.
    if(_operations->has_operation(vertex.name())) {
        assert(!vertex.operation());
        OperationPtr const& operation(_operations->operation(
            vertex.name()));
        vertex.set_operation(operation);

        assert(vertex.result_types().empty());

        // http://en.cppreference.com/w/cpp/types/common_type
        // There are vertex.expressions().size() ResultType instances on the
        // stack. Given these and the operation, calculate a ResultType
        // instance for the result of this expression. It is possible that
        // there are not enough arguments provided. In that case the
        // calculation of the result type may fail. Validation will pick that
        // up.

        // Get the result types from all argument expressions provided.
        std::vector<ResultType> argument_types;
        for(size_t i = 0; i < vertex.expressions().size(); ++i) {
            assert(!_stack.empty());
            argument_types.push_back(_stack.top());
            _stack.pop();
        }

        ResultTypes result_types(vertex.expressions().size());
        if(vertex.expressions().size() == operation->arity()) {
            // Calculate result type for each result.
            for(size_t i = 0; i < operation->results().size(); ++i) {
                ResultType result_type = operation->result_type(i,
                    argument_types);
                _stack.push(result_type);
                result_types.push_back(result_type);
            }
        }
        vertex.set_result_types(result_types);
    }
}


void AnnotateVisitor::Visit(
    NameVertex& vertex)
{
    ResultType result_type;

    // If the symbol is defined, retrieve the value from the symbol table.
    if(_symbol_table.has_value(vertex.name())) {
        result_type = _symbol_table.value(vertex.name());
    }

    // Push the value onto the stack, even if it is undefined.
    _stack.push(result_type);
    vertex.add_result_type(result_type);
}


void AnnotateVisitor::Visit(
    ScriptVertex& vertex)
{
    _symbol_table.push_scope();
    Visitor::Visit(vertex);
    _symbol_table.pop_scope();
}


// void AnnotateVisitor::Visit(
//     SubscriptVertex& /* vertex */)
// {
//     // TODO
//     // switch(_mode) {
//     //     case Using: {
//     //         // The result type of a subscript expression is the same as the
//     //         // result type of the main expression.
//     //         assert(vertex.result_types().empty());
//     //         vertex.set_result_types(vertex.expression()->result_types());
//     //         break;
//     //     }
//     //     case Defining: {
//     //         break;
//     //     }
//     // }
// }

} // namespace ranally
