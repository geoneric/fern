#include "ranally/interpreter/execute_visitor.h"
#include "ranally/ast/core/vertices.h"
#include "ranally/feature/scalar_attribute.h"
#include "ranally/operation/core/attribute_argument.h"


namespace ranally {

ExecuteVisitor::ExecuteVisitor(
    OperationsPtr const& operations)

    : _operations(operations),
      _stack(),
      _symbol_table()

{
   assert(operations);
   _symbol_table.push_scope();
}


ExecuteVisitor::~ExecuteVisitor()
{
   _symbol_table.pop_scope();
}


std::stack<std::shared_ptr<Argument>> const&
ExecuteVisitor::stack() const
{
    return _stack;
}


void ExecuteVisitor::clear_stack()
{
    _stack = std::stack<std::shared_ptr<Argument>>();
}


SymbolTable<std::shared_ptr<Argument>> const&
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


// TODO Connect this attribute to a global script feature.
#define VISIT_NUMBER_VERTEX(                                                   \
    type)                                                                      \
void ExecuteVisitor::Visit(                                                    \
    NumberVertex<type>& vertex)                                                \
{                                                                              \
    _stack.push(                                                               \
        std::shared_ptr<Argument>(new AttributeArgument(                       \
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

    Operation const& operation(*_operations->operation(vertex.name()));
    assert(_stack.size() >= operation.arity());

    std::vector<std::shared_ptr<Argument>> arguments;
    for(size_t i = 0; i < vertex.expressions().size(); ++i) {
        assert(!_stack.empty());
        arguments.push_back(_stack.top());
        _stack.pop();
    }

    for(auto result: operation.execute(arguments)) {
        _stack.push(result);
    }
}


void ExecuteVisitor::Visit(
    ModuleVertex& vertex)
{
    Visitor::Visit(vertex);
}


void ExecuteVisitor::Visit(
    StringVertex& vertex)
{
    // TODO Connect this attribute to a global script feature.
    // Turn the string constant into an attribute with a global domain and a
    // single value.
    _stack.push(
        std::shared_ptr<Argument>(new AttributeArgument(
            std::shared_ptr<Attribute>(new ScalarAttribute<String>(
                std::make_shared<ScalarDomain>(),
                std::make_shared<ScalarValue<String>>(vertex.value()))))));
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
