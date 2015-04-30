// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/ast/visitor/identify_visitor.h"
#include "fern/core/string.h"
#include "fern/language/ast/core/vertices.h"


namespace fern {
namespace language {

IdentifyVisitor::IdentifyVisitor()

    : AstVisitor(),
      _mode(Mode::Using)

{
}


void IdentifyVisitor::Visit(
    AssignmentVertex& vertex)
{
    // Order matters. First handle the uses, then the
    // definitions. Otherwise the use in the expression might be connected
    // to the definition in the same statement, in a = a + b, for example.

    // - Configure visitor, using names.
    // - Visit expression.
    _mode = Mode::Using;
    vertex.expression()->Accept(*this);

    // - Configure visitor, defining names.
    // - Visit target.
    _mode = Mode::Defining;
    vertex.target()->Accept(*this);

    // Reset mode! Only in assignments is the mode temporarely set to defining.
    _mode = Mode::Using;
}


void IdentifyVisitor::Visit(
    FunctionCallVertex& vertex)
{
    visit_expressions(vertex.expressions());
}


void IdentifyVisitor::Visit(
    NameVertex& vertex)
{
    switch(_mode) {
        case Mode::Using: {
            // Using a name, connect it to the definition.
            assert(vertex.definitions().empty());

            if(_symbol_table.has_value(vertex.name())) {
                // TODO: A name can have multiple definitions. Deeper
                // scopes can update identifiers in upper scopes, for
                // example in an if-block.
                // Search for all definitions in the current and
                // previous deeper scopes. Instead of using the most recent
                // definition, we want connections with all possible relevant
                // definitions. Only at runtime do we know exactly where a
                // identifier is defined. Also, depending on the data type,
                // data may be defined partly at one location and partly
                // at another.
                // Definitions don't overwrite each other, per se. In
                // case of a definition in an if-block, it depends on the
                // condition. Also, data type is relevant, as described
                // above.
                // A name is available if it is defined in the current
                // or higher scope. All current and higher definitions are
                // relevant here.
                NameVertex* definition =
                    _symbol_table.value(vertex.name());
                vertex.add_definition(definition);
                definition->add_use(&vertex);
            }

            break;
        }
        case Mode::Defining: {
            // Defining a name, add it to the symbol table.
            assert(vertex.definitions().empty());
            vertex.add_definition(&vertex);
            _symbol_table.add_value(vertex.name(), &vertex);
            break;
        }
    }
}


void IdentifyVisitor::Visit(
    OperatorVertex& vertex)
{
    visit_expressions(vertex.expressions());
}


// Default behavior of AstVisitor already.
// void IdentifyVisitor::Visit(
//     SubscriptVertex& vertex)
// {
//     vertex.expression()->Accept(*this);
//     vertex.selection()->Accept(*this);
// }


void IdentifyVisitor::Visit(
    ModuleVertex& vertex)
{
    assert(_symbol_table.empty());
    _symbol_table.push_scope();
    visit_statements(vertex.scope()->statements());
    _symbol_table.pop_scope();
    assert(_symbol_table.empty());
}


void IdentifyVisitor::Visit(
    IfVertex& vertex)
{
    vertex.condition()->Accept(*this);

    _symbol_table.push_scope();
    visit_statements(vertex.true_scope()->statements());
    _symbol_table.pop_scope();

    if(vertex.false_scope()) {
        _symbol_table.push_scope();
        visit_statements(vertex.false_scope()->statements());
        _symbol_table.pop_scope();
    }
}


void IdentifyVisitor::Visit(
    WhileVertex& vertex)
{
    vertex.condition()->Accept(*this);

    _symbol_table.push_scope();
    visit_statements(vertex.true_scope()->statements());
    _symbol_table.pop_scope();

    if(vertex.false_scope()) {
        _symbol_table.push_scope();
        visit_statements(vertex.false_scope()->statements());
        _symbol_table.pop_scope();
    }
}


SymbolTable<NameVertex*> const& IdentifyVisitor::symbol_table() const
{
    return _symbol_table;
}

} // namespace language
} // namespace fern
