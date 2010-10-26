#include <iostream>
#include <boost/foreach.hpp>
#include <boost/format.hpp>

#include "AssignmentVertex.h"
#include "FunctionVertex.h"
#include "NameVertex.h"
#include "NumberVertex.h"
#include "ScriptVertex.h"
#include "ScriptVisitor.h"
#include "StatementVertex.h"
#include "StringVertex.h"



namespace ranally {

ScriptVisitor::ScriptVisitor()
{
}



ScriptVisitor::~ScriptVisitor()
{
}



UnicodeString ScriptVisitor::Visit(
  AssignmentVertex& vertex)
{
  ExpressionVertices const& targets = vertex.targets();
  assert(targets.size() == 1);

  ExpressionVertices const& expressions = vertex.expressions();
  assert(expressions.size() == 1);

  UnicodeString result;
  result += targets[0]->Accept(*this);
  result += " = ";
  result += expressions[0]->Accept(*this);

  return result;
}



UnicodeString ScriptVisitor::Visit(
  FunctionVertex& vertex)
{
  UnicodeString result = vertex.name();
  result += "(";

  std::vector<UnicodeString> scripts;
  BOOST_FOREACH(boost::shared_ptr<ranally::ExpressionVertex> expressionVertex,
    vertex.expressions()) {
    scripts.push_back(expressionVertex->Accept(*this));
  }

  if(!scripts.empty()) {
    result += scripts[0];

    for(size_t i = 1; i < scripts.size(); ++i) {
      result += ", " + scripts[i];
    }
  }

  result += ")";
  return result;
}



UnicodeString ScriptVisitor::Visit(
  SyntaxVertex&)
{
  assert(false);
}



UnicodeString ScriptVisitor::Visit(
  ScriptVertex& vertex)
{
  UnicodeString result;

  BOOST_FOREACH(boost::shared_ptr<ranally::StatementVertex> statementVertex,
    vertex.statements()) {
    result += statementVertex->Accept(*this);
  }

  return result;
}



UnicodeString ScriptVisitor::Visit(
  StringVertex& vertex)
{
  return "\"" + vertex.string() + "\"";
}



UnicodeString ScriptVisitor::Visit(
  NameVertex& vertex)
{
  return vertex.name();
}



UnicodeString ScriptVisitor::Visit(
  NumberVertex<int>& vertex)
{
  return UnicodeString((boost::format("%1%") % vertex.value()).str().c_str());
}



UnicodeString ScriptVisitor::Visit(
  NumberVertex<long long>& /* vertex */)
{
  return "long long";
}



UnicodeString ScriptVisitor::Visit(
  NumberVertex<double>& /* vertex */)
{
  return "double";
}

} // namespace ranally

