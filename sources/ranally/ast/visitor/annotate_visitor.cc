#include "ranally/ast/visitor/annotate_visitor.h"
#include "ranally/core/string.h"
#include "ranally/operation/core/result.h"
#include "ranally/operation/core/type_traits.h"
#include "ranally/ast/core/vertices.h"


namespace ranally {

AnnotateVisitor::AnnotateVisitor(
    ranally::OperationsPtr const& operations)

    : Visitor(),
      _stack(),
      _symbol_table(),
      _operations(operations)

{
    assert(_operations);
    _symbol_table.push_scope();
}


AnnotateVisitor::~AnnotateVisitor()
{
    _symbol_table.pop_scope();
}


std::stack<ResultType> const& AnnotateVisitor::stack() const
{
    return _stack;
}


void AnnotateVisitor::clear_stack()
{
    _stack = std::stack<ResultType>();
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

    // Propagate the result types from the expression to the target.
    ExpressionVertex const& expression(*vertex.expression());
    ExpressionVertex& target(*vertex.target());
    assert(target.result_types().empty());
    assert(expression.result_types().size() == 1);
    target.set_result_types(expression.result_types());
    // vertex.target()->Accept(*this);
}


#define VISIT_NUMBER_VERTEX(                                                   \
    type)                                                                      \
void AnnotateVisitor::Visit(                                                   \
    NumberVertex<type>& vertex)                                                \
{                                                                              \
    assert(vertex.result_types().empty());                                     \
    ResultType result_type(DataTypes::SCALAR, TypeTraits<type>::value_types);  \
    _stack.push(result_type);                                                  \
    vertex.add_result_type(result_type);                                       \
    assert(vertex.result_types().size() == 1);                                 \
}

VISIT_NUMBER_VERTICES(VISIT_NUMBER_VERTEX)

#undef VISIT_NUMBER_VERTEX


void AnnotateVisitor::Visit(
    OperationVertex& vertex)
{
    assert(vertex.result_types().empty());

    // Depth first, visit the children first.
    Visitor::Visit(vertex);

    // Get the result types from all argument expressions provided.
    std::vector<ResultType> argument_types;
    for(size_t i = 0; i < vertex.expressions().size(); ++i) {
        assert(!_stack.empty());
        argument_types.push_back(_stack.top());
        _stack.pop();
    }

    ResultTypes result_types(1);

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

        if(vertex.expressions().size() == operation->arity()) {
            // Calculate result type for each result.
            // TODO Update for multiple results.
            for(size_t i = 0; i < 1; ++i) {
                ResultType result_type = operation->result_type(i,
                    argument_types);
                result_types[i] = result_type;
            }
        }
    }

    for(auto result_type: result_types) {
        _stack.push(result_type);
    }
    vertex.set_result_types(result_types);

    assert(vertex.result_types().size() == 1);
}


void AnnotateVisitor::Visit(
    NameVertex& vertex)
{
    assert(vertex.result_types().empty());

    ResultType result_type;

    // If the symbol is defined, retrieve the value from the symbol table.
    if(_symbol_table.has_value(vertex.name())) {
        result_type = _symbol_table.value(vertex.name());
    }

    // Push the value onto the stack, even if it is undefined.
    _stack.push(result_type);
    vertex.add_result_type(result_type);

    assert(vertex.result_types().size() == 1);
}


void AnnotateVisitor::Visit(
    ScriptVertex& vertex)
{
    Visitor::Visit(vertex);
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
