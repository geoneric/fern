#include "Ranally/Language/IdentifyVisitor.h"
#include <boost/foreach.hpp>
#include "Ranally/Language/Vertices.h"
#include "Ranally/Util/String.h"


namespace ranally {

IdentifyVisitor::IdentifyVisitor()

    : Visitor(),
      _mode(Using)

{
}


IdentifyVisitor::~IdentifyVisitor()
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
    _mode = Using;
    vertex.expression()->Accept(*this);

    // - Configure visitor, defining names.
    // - Visit target.
    _mode = Defining;
    vertex.target()->Accept(*this);

    // Reset mode! Only in assignments is the mode temporarely set to defining.
    _mode = Using;
}


void IdentifyVisitor::Visit(
    FunctionVertex& vertex)
{
    visitExpressions(vertex.expressions());
}


void IdentifyVisitor::Visit(
    NameVertex& vertex)
{
    switch(_mode) {
        case Using: {
            // Using a name, connect it to the definition.
            assert(vertex.definitions().empty());

            if(_symbolTable.hasDefinition(vertex.name())) {
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
                NameVertex* definition = _symbolTable.definition(vertex.name());
                vertex.addDefinition(definition);
                definition->addUse(&vertex);
            }

            break;
        }
        case Defining: {
            // Defining a name, add it to the symbol table.
            assert(vertex.definitions().empty());
            vertex.addDefinition(&vertex);
            _symbolTable.addDefinition(&vertex);
            break;
        }
    }
}


void IdentifyVisitor::Visit(
    OperatorVertex& vertex)
{
    visitExpressions(vertex.expressions());
}


void IdentifyVisitor::Visit(
    ScriptVertex& vertex)
{
    assert(_symbolTable.empty());
    _symbolTable.pushScope();
    visitStatements(vertex.statements());
    _symbolTable.popScope();
    assert(_symbolTable.empty());
}


void IdentifyVisitor::Visit(
    IfVertex& vertex)
{
    vertex.condition()->Accept(*this);

    assert(!vertex.trueStatements().empty());
    _symbolTable.pushScope();
    visitStatements(vertex.trueStatements());
    _symbolTable.popScope();

    if(!vertex.falseStatements().empty()) {
        _symbolTable.pushScope();
        visitStatements(vertex.falseStatements());
        _symbolTable.popScope();
    }
}


void IdentifyVisitor::Visit(
    WhileVertex& vertex)
{
    vertex.condition()->Accept(*this);

    assert(!vertex.trueStatements().empty());
    _symbolTable.pushScope();
    visitStatements(vertex.trueStatements());
    _symbolTable.popScope();

    if(!vertex.falseStatements().empty()) {
        _symbolTable.pushScope();
        visitStatements(vertex.falseStatements());
        _symbolTable.popScope();
    }
}


SymbolTable const& IdentifyVisitor::symbolTable() const
{
    return _symbolTable;
}

} // namespace ranally
