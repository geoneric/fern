#include "ranally/interpreter/execute_visitor.h"
#include "ranally/ast/core/vertices.h"
#include "ranally/feature/scalar_attribute.h"
#include "ranally/interpreter/attribute_value.h"


namespace ranally {

//! Execute operation \a name given \a arguments, and return the results.
/*!
  \param     name Name of operation to execute.
  \param     arguments Arguments to pass to the operation.
  \return    Zero or more results calculated by the operation.
*/
std::vector<std::shared_ptr<interpreter::Value>> execute_operation(
    String const& name,
    std::vector<std::shared_ptr<interpreter::Value>> arguments)
{
    std::vector<std::shared_ptr<interpreter::Value>> results;

    // TODO


    // Given
    // - name
    // - arguments
    // - result
    // Instantiate operation class that translates the arguments to the results.
    //
    // -> Create Operation hierarchy:
    //     Operation
    //     -> Unary Operation
    //     -> Binary Operation
    //     -> NAry Operation
    //     Or only one Operation class that all operations inherit. Per
    //     ConcreteOperation the operation is executed.


    return results;
}


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
    // Let the argument expressions execute themselves. The resulting values
    // end up at the top of the stack.
    visit_expressions(vertex.expressions());

    std::vector<std::shared_ptr<interpreter::Value>> arguments;
    for(size_t i = 0; i < vertex.expressions().size(); ++i) {
        assert(!_stack.empty());
        arguments.push_back(_stack.top());
        _stack.pop();
    }

    assert(vertex.operation());
    Operation const& operation(*vertex.operation());
    assert(_stack.size() >= operation.arity());

    std::vector<std::shared_ptr<interpreter::Value>> results =
        execute_operation(operation.name(), arguments);

    for(auto result: results) {
        _stack.push(result);
    }
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
