#include <boost/foreach.hpp>

#include "IdentifyVisitor.h"
#include "Vertices.h"



namespace ranally {
namespace language {

IdentifyVisitor::IdentifyVisitor()

  : _mode(Using)

{
}



IdentifyVisitor::~IdentifyVisitor()
{
}



void IdentifyVisitor::visitStatements(
  StatementVertices const& statements)
{
  BOOST_FOREACH(boost::shared_ptr<ranally::StatementVertex> statementVertex,
    statements) {
    statementVertex->Accept(*this);
  }
}



void IdentifyVisitor::visitExpressions(
  ExpressionVertices const& expressions)
{
  BOOST_FOREACH(boost::shared_ptr<ranally::ExpressionVertex> expressionVertex,
    expressions) {
    expressionVertex->Accept(*this);
  }
}



void IdentifyVisitor::Visit(
  AssignmentVertex& vertex)
{
  // - Configure visitor, defining names.
  // - Visit targets.
  _mode = Defining;
  ExpressionVertices const& targets = vertex.targets();
  assert(targets.size() == 1);
  targets[0]->Accept(*this);

  // - Configure visitor, using names.
  // - Visit expressions.
  _mode = Using;
  ExpressionVertices const& expressions = vertex.expressions();
  assert(expressions.size() == 1);
  expressions[0]->Accept(*this);
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
      // TODO
      break;
    }
    case Defining: {
      // Defining a name, add it to the symbol table.
      vertex.setDefinition(&vertex);
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
  _symbolTable.pushScope();
  visitStatements(vertex.statements());
  _symbolTable.popScope();
}



SymbolTable const& IdentifyVisitor::symbolTable() const
{
  return _symbolTable;
}

} // namespace language
} // namespace ranally

