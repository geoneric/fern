#include "Ranally/Language/ThreadVisitor.h"
#include <boost/foreach.hpp>
#include "Ranally/Language/Vertices.h"



namespace ranally {
namespace language {

ThreadVisitor::ThreadVisitor()

  : _lastVertex(0)

{
}



ThreadVisitor::~ThreadVisitor()
{
}



void ThreadVisitor::visitStatements(
  StatementVertices const& statements)
{
  BOOST_FOREACH(boost::shared_ptr<StatementVertex> statementVertex,
    statements) {
    statementVertex->Accept(*this);
  }
}



void ThreadVisitor::visitExpressions(
  ExpressionVertices const& expressions)
{
  BOOST_FOREACH(boost::shared_ptr<ExpressionVertex> expressionVertex,
    expressions) {
    expressionVertex->Accept(*this);
  }
}



void ThreadVisitor::Visit(
  AssignmentVertex& vertex)
{
  ExpressionVertices const& expressions = vertex.expressions();
  assert(expressions.size() == 1);
  expressions[0]->Accept(*this);

  ExpressionVertices const& targets = vertex.targets();
  assert(targets.size() == 1);
  targets[0]->Accept(*this);

  assert(_lastVertex);
  _lastVertex->addSuccessor(&vertex);
  _lastVertex = &vertex;
}



void ThreadVisitor::Visit(
  FunctionVertex& vertex)
{
  visitExpressions(vertex.expressions());

  assert(_lastVertex);
  _lastVertex->addSuccessor(&vertex);
  _lastVertex = &vertex;
}



void ThreadVisitor::Visit(
  OperatorVertex& vertex)
{
  visitExpressions(vertex.expressions());

  assert(_lastVertex);
  _lastVertex->addSuccessor(&vertex);
  _lastVertex = &vertex;
}



void ThreadVisitor::Visit(
  SyntaxVertex&)
{
  assert(false);
}



void ThreadVisitor::Visit(
  ScriptVertex& vertex)
{
  _lastVertex = &vertex;
  visitStatements(vertex.statements());
  assert(_lastVertex);
  _lastVertex->addSuccessor(&vertex);
}



void ThreadVisitor::Visit(
  StringVertex& vertex)
{
  assert(_lastVertex);
  _lastVertex->addSuccessor(&vertex);
  _lastVertex = &vertex;
}



void ThreadVisitor::Visit(
  NameVertex& vertex)
{
  assert(_lastVertex);
  _lastVertex->addSuccessor(&vertex);
  _lastVertex = &vertex;
}



template<typename T>
void ThreadVisitor::Visit(
  NumberVertex<T>& vertex)
{
  assert(_lastVertex);
  _lastVertex->addSuccessor(&vertex);
  _lastVertex = &vertex;
}



void ThreadVisitor::Visit(
  NumberVertex<int8_t>& vertex)
{
  Visit<int8_t>(vertex);
}



void ThreadVisitor::Visit(
  NumberVertex<int16_t>& vertex)
{
  Visit<int16_t>(vertex);
}



void ThreadVisitor::Visit(
  NumberVertex<int32_t>& vertex)
{
  Visit<int32_t>(vertex);
}



void ThreadVisitor::Visit(
  NumberVertex<int64_t>& vertex)
{
  Visit<int64_t>(vertex);
}



void ThreadVisitor::Visit(
  NumberVertex<uint8_t>& vertex)
{
  Visit<uint8_t>(vertex);
}



void ThreadVisitor::Visit(
  NumberVertex<uint16_t>& vertex)
{
  Visit<uint16_t>(vertex);
}



void ThreadVisitor::Visit(
  NumberVertex<uint32_t>& vertex)
{
  Visit<uint32_t>(vertex);
}



void ThreadVisitor::Visit(
  NumberVertex<uint64_t>& vertex)
{
  Visit<uint64_t>(vertex);
}



void ThreadVisitor::Visit(
  NumberVertex<float>& vertex)
{
  Visit<float>(vertex);
}



void ThreadVisitor::Visit(
  NumberVertex<double>& vertex)
{
  Visit<double>(vertex);
}



void ThreadVisitor::Visit(
  IfVertex& vertex)
{
  // First let the condition thread itself.
  vertex.condition()->Accept(*this);

  // Now we must get the control.
  assert(_lastVertex);
  _lastVertex->addSuccessor(&vertex);

  // Let the true and false block thread themselves.
  _lastVertex = &vertex;
  assert(!vertex.trueStatements().empty());
  visitStatements(vertex.trueStatements());
  _lastVertex->addSuccessor(&vertex);

  if(!vertex.falseStatements().empty()) {
    _lastVertex = &vertex;
    visitStatements(vertex.falseStatements());
    _lastVertex->addSuccessor(&vertex);
  }

  _lastVertex = &vertex;
}



void ThreadVisitor::Visit(
  WhileVertex& /* vertex */)
{
  // TODO
  assert(false);
}

} // namespace language
} // namespace ranally

