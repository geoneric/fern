#include "ranally/language/execute_visitor.h"
#include "ranally/language/vertices.h"


namespace ranally {

void ExecuteVisitor::Visit(
    AssignmentVertex& /* vertex */)
{
    // Let the expression execute itself.
    // Store the result in a scoped symbol table for later reference.
    // Update scope at correct moments in other visit functions.

    // _symbol_table.add_value(vertex.target()->name(), value);
}



void ExecuteVisitor::Visit(
    IfVertex& /* vertex */)
{
    // Evaluate condition.
    // TODO
}


void ExecuteVisitor::Visit(
    WhileVertex& /* vertex */)
{
    // Evaluate condition.
    // TODO
}


void ExecuteVisitor::Visit(
    NameVertex& /* vertex */)
{
    // Retrieve the value from the symbol table and push it onto the stack.
    // _stack.push(_symbol_table.value(vertex.name()));
}


#define VISIT_NUMBER_VERTEX(                                                   \
    type)                                                                      \
void ExecuteVisitor::Visit(                                                    \
    NumberVertex<type>& vertex)                                                \
{                                                                              \
    _stack.push<type>(vertex.value());                                         \
}

VISIT_NUMBER_VERTEX(int8_t  )
VISIT_NUMBER_VERTEX(int16_t )
VISIT_NUMBER_VERTEX(int32_t )
VISIT_NUMBER_VERTEX(int64_t )
VISIT_NUMBER_VERTEX(uint8_t )
VISIT_NUMBER_VERTEX(uint16_t)
VISIT_NUMBER_VERTEX(uint32_t)
VISIT_NUMBER_VERTEX(uint64_t)
VISIT_NUMBER_VERTEX(float   )
VISIT_NUMBER_VERTEX(double  )

#undef VISIT_NUMBER_VERTEX


void ExecuteVisitor::Visit(
    OperationVertex& vertex)
{
    // Maybe create a hierarchy of execution blocks that share the same
    // base class, but are templated on the argument types, result types and
    // operation type. (It also includes control flow blocks.) Each execution
    // block is capable of returning its results in the correct types. Maybe
    // each execution block is a functor.
    // Arguments of operations then become execution blocks that can execute
    // themselves and return a result.
    // Can we use this idea for a compiler too?

    // Let the argument expressions execute themselves. The resulting values
    // end up at the top of the stack.
    visit_expressions(vertex.expressions());

    Operation const& operation(*vertex.operation());
    assert(_stack.size() >= operation.arity());

    // We have everything we need: operation, arguments.
    // Given the properties of the operation:
    // TODO Try one out, create a test and see how it works.

    if(operation.name() == "Add") {
        assert(operation.arity() == 2);
        assert(vertex.data_type(0) == DataTypes::SCALAR);
        assert(vertex.data_type(1) == DataTypes::SCALAR);
        assert(vertex.value_type(0) == ValueTypes::FLOAT64);
        assert(vertex.value_type(1) == ValueTypes::FLOAT64);

        double result = _stack.top<double>(); _stack.pop();
        result += _stack.top<double>(); _stack.pop();
        _stack.push(result);
    }
}


void ExecuteVisitor::Visit(
    StringVertex& /* vertex */)
{
    // Read attribute value from dataset and store on stack.
    // _stack.push(read(vertex.value());
}


void ExecuteVisitor::Visit(
    SubscriptVertex& /* vertex */)
{
    // Evaluate expression and determine value.
    // The selection expression must be evaluated given/on the value.
    // We need a set of operations that make selections.
    // Evaluating selection expressions requires a different executor. The
    // expressions have a different meaning in the context of subscripts.
    // TODO
}

} // namespace ranally
