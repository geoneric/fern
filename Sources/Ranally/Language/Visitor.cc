#include "Ranally/Language/Visitor.h"
#include <boost/foreach.hpp>
#include "Ranally/Language/ExpressionVertex.h"
#include "Ranally/Language/StatementVertex.h"



namespace ranally {
namespace language {

Visitor::Visitor()
{
}



Visitor::~Visitor()
{
}



void Visitor::visitStatements(
  StatementVertices const& statements)
{
  BOOST_FOREACH(boost::shared_ptr<StatementVertex> statementVertex,
    statements) {
    statementVertex->Accept(*this);
  }
}



void Visitor::visitExpressions(
  ExpressionVertices const& expressions)
{
  BOOST_FOREACH(boost::shared_ptr<ExpressionVertex> expressionVertex,
    expressions) {
    expressionVertex->Accept(*this);
  }
}



void Visitor::Visit(
  AssignmentVertex& /* vertex */)
{
}



void Visitor::Visit(
  FunctionVertex& /* vertex */)
{
}



void Visitor::Visit(
  IfVertex& /* vertex */)
{
}



void Visitor::Visit(
  NameVertex& /* vertex */)
{
}



void Visitor::Visit(
  language::NumberVertex<int8_t>& /* vertex */)
{
}



void Visitor::Visit(
  language::NumberVertex<int16_t>& /* vertex */)
{
}



void Visitor::Visit(
  language::NumberVertex<int32_t>& /* vertex */)
{
}



void Visitor::Visit(
  language::NumberVertex<int64_t>& /* vertex */)
{
}



void Visitor::Visit(
  language::NumberVertex<uint8_t>& /* vertex */)
{
}



void Visitor::Visit(
  language::NumberVertex<uint16_t>& /* vertex */)
{
}



void Visitor::Visit(
  language::NumberVertex<uint32_t>& /* vertex */)
{
}



void Visitor::Visit(
  language::NumberVertex<uint64_t>& /* vertex */)
{
}



void Visitor::Visit(
  language::NumberVertex<float>& /* vertex */)
{
}



void Visitor::Visit(
  language::NumberVertex<double>& /* vertex */)
{
}



void Visitor::Visit(
  OperatorVertex& /* vertex */)
{
}



void Visitor::Visit(
  ScriptVertex& /* vertex */)
{
}



void Visitor::Visit(
  StringVertex& /* vertex */)
{
}



void Visitor::Visit(
  WhileVertex& /* vertex */)
{
}


} // namespace language
} // namespace ranally

