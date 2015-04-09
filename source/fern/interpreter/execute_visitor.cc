// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/interpreter/execute_visitor.h"
#include "fern/ast/core/vertices.h"
#include "fern/ast/visitor/io_visitor.h"
#include "fern/feature/core/constant_attribute.h"
#include "fern/operation/core/attribute_argument.h"
#include "fern/interpreter/data_sources.h"
#include "fern/interpreter/data_syncs.h"


namespace fern {

ExecuteVisitor::ExecuteVisitor(
    OperationsPtr const& operations)

    : AstVisitor(),
      _operations(operations),
      _stack(),
      _symbol_table(),
      _data_source_symbol_table(),
      _data_sync_symbol_table(),
      _outputs()

{
   assert(operations);
   _symbol_table.push_scope();
   _data_source_symbol_table.push_scope();
   _data_sync_symbol_table.push_scope();
}


ExecuteVisitor::~ExecuteVisitor()
{
   _symbol_table.pop_scope();
   _data_source_symbol_table.pop_scope();
   _data_sync_symbol_table.pop_scope();
}


void ExecuteVisitor::set_data_source_symbols(
    SymbolTable<std::shared_ptr<DataSource>> const& symbol_table)
{
    assert(_data_source_symbol_table.scope_level() == 1u);
    _data_source_symbol_table.clear_scope();
    assert(_symbol_table.scope_level() == 1u);

    if(!symbol_table.empty()) {
        assert(symbol_table.scope_level() == 1u);

        for(auto const& pair: symbol_table.scope(1u)) {
            assert(!_data_source_symbol_table.has_value(pair.first));
            if(_symbol_table.has_value(pair.first)) {
                _symbol_table.erase_value(pair.first);
            }
            _data_source_symbol_table.add_value(pair.first, pair.second);
        }
    }
}


void ExecuteVisitor::set_data_sync_symbols(
    SymbolTable<std::shared_ptr<DataSync>> const& symbol_table)
{
    assert(_data_sync_symbol_table.scope_level() == 1u);
    _data_sync_symbol_table.clear_scope();

    if(!symbol_table.empty()) {
        assert(symbol_table.scope_level() == 1u);
        _data_sync_symbol_table = symbol_table;
    }
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
    assert(!_stack.empty());

    // Assume the target expression is a NameVertex (it should, for now).
    assert(dynamic_cast<NameVertex const*>(vertex.target().get()));

    // If the target vertex is an output, and if the output is mentioned in
    // the data sync symbol table, then write out the result.
    if(std::find(_outputs.begin(), _outputs.end(), vertex.target().get()) !=
            _outputs.end()) {
        if(_data_sync_symbol_table.has_value(vertex.target()->name())) {
            _data_sync_symbol_table.value(vertex.target()->name())->write(
                *_stack.top());
        }
    }

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
    if(!_symbol_table.has_value(vertex.name())) {
        // We must have a data source for this input variable.
        assert(_data_source_symbol_table.has_value(vertex.name()));

        // Add the result of the read to the symbol table, so later
        // references to the same symbol can use the data read, instead of
        // reading the same data multiple times.
        _symbol_table.add_value(vertex.name(),
            _data_source_symbol_table.value(vertex.name())->read());
    }

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
        std::shared_ptr<Argument>(std::make_shared<AttributeArgument>(         \
            std::make_shared<ConstantAttribute<type>>(vertex.value()))));      \
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
        // Arguments are popped from the top of the stack, in reverse order.
        arguments.insert(arguments.begin(), _stack.top());
        _stack.pop();
    }

    for(auto result: operation.execute(arguments)) {
        _stack.push(result);
    }
}


void ExecuteVisitor::Visit(
    ModuleVertex& vertex)
{
    // Determine inputs and outputs of the module.
    IOVisitor visitor;
    vertex.Accept(visitor);
    _outputs = visitor.outputs();

    AstVisitor::Visit(vertex);
}


void ExecuteVisitor::Visit(
    StringVertex& vertex)
{
    // TODO Connect this attribute to a global script-feature.
    // Turn the string constant into an attribute with a global domain and a
    // single value.
    _stack.push(
        std::shared_ptr<Argument>(std::make_shared<AttributeArgument>(
            std::make_shared<ConstantAttribute<String>>(vertex.value()))));
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

} // namespace fern
