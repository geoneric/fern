#include "ranally/interpreter/execute_visitor.h"
#include "ranally/language/vertices.h"
#include "ranally/feature/scalar_attribute.h"
#include "ranally/interpreter/attribute_value.h"


namespace ranally {

ExecuteVisitor::ExecuteVisitor()

    : _stack(),
      _symbol_table()

{
   _symbol_table.push_scope();
}


ExecuteVisitor::~ExecuteVisitor()
{
   _symbol_table.pop_scope();
}


std::stack<std::shared_ptr<interpreter::Value>> const&
ExecuteVisitor::stack() const
{
    return _stack;
}


void ExecuteVisitor::clear_stack()
{
    _stack = std::stack<std::shared_ptr<interpreter::Value>>();
}


SymbolTable<std::shared_ptr<interpreter::Value>> const&
ExecuteVisitor::symbol_table() const
{
    return _symbol_table;
}


void ExecuteVisitor::Visit(
    AssignmentVertex& vertex)
{
    // Let the source expression execute itself, leaving the result(s) on the
    // stack.
    vertex.expression()->Accept(*this);

    // Assume the target expression is a NameVertex (it should, for now).
    assert(dynamic_cast<NameVertex const*>(vertex.target().get()));

    // Store the result in a scoped symbol table for later reference.
    // Update scope at correct moments in other visit functions.
    _symbol_table.add_value(vertex.target()->name(), _stack.top());
    _stack.pop();
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
    NameVertex& vertex)
{
    // Retrieve the value from the symbol table and push it onto the stack.
    assert(_symbol_table.has_value(vertex.name()));
    _stack.push(_symbol_table.value(vertex.name()));
}


#define VISIT_NUMBER_VERTEX(                                                   \
    type)                                                                      \
void ExecuteVisitor::Visit(                                                    \
    NumberVertex<type>& vertex)                                                \
{                                                                              \
    _stack.push(                                                               \
        std::shared_ptr<interpreter::Value>(new interpreter::AttributeValue(   \
            std::shared_ptr<Attribute>(new ScalarAttribute<type>(              \
                std::make_shared<ScalarDomain>(),                              \
                std::make_shared<ScalarValue<type>>(vertex.value()))))));      \
}

VISIT_NUMBER_VERTICES(VISIT_NUMBER_VERTEX)

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

    assert(vertex.operation());
    Operation const& operation(*vertex.operation());
    assert(_stack.size() >= operation.arity());

    for(size_t i = 1; i < vertex.expressions().size(); ++i) {
        _stack.pop();
    }

    // We have everything we need: operation, arguments.
    // Given the properties of the operation:
    // TODO Try one out, create a test and see how it works.

    // if(operation.name() == "add") {
    //     assert(operation.arity() == 2);
    //     assert(vertex.data_type(0) == DataTypes::SCALAR);
    //     assert(vertex.data_type(1) == DataTypes::SCALAR);
    //     assert(vertex.value_type(0) == ValueTypes::INT64);
    //     assert(vertex.value_type(1) == ValueTypes::INT64);

    //     int64_t result = _stack.top<int64_t>(); _stack.pop();
    //     result += _stack.top<int64_t>(); _stack.pop();
    //     _stack.push(result);
    // }
}


void ExecuteVisitor::Visit(
    ScriptVertex& vertex)
{
    Visitor::Visit(vertex);
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
