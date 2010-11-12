// #include <iostream>
#include <boost/foreach.hpp>
#include <boost/format.hpp>

// #include "dev_UnicodeUtils.h"

#include "AssignmentVertex.h"
#include "FunctionVertex.h"
#include "IfVertex.h"
#include "NameVertex.h"
#include "NumberVertex.h"
#include "OperatorVertex.h"
#include "ScriptVertex.h"
#include "DotVisitor.h"
#include "StatementVertex.h"
#include "StringVertex.h"
#include "WhileVertex.h"



namespace ranally {

DotVisitor::DotVisitor()

  : _tabSize(2),
    _indentLevel(0)

{
}



DotVisitor::~DotVisitor()
{
}



UnicodeString DotVisitor::indent(
  UnicodeString const& statement)
{
  UnicodeString indentation = std::string(_indentLevel * _tabSize, ' ').c_str();
  return indentation + statement;
}



UnicodeString DotVisitor::visitStatements(
  StatementVertices const& statements)
{
  UnicodeString result;

  BOOST_FOREACH(boost::shared_ptr<ranally::StatementVertex> statementVertex,
    statements) {
    result += statementVertex->Accept(*this);
  }

  return result;
}



UnicodeString DotVisitor::visitExpressions(
  ExpressionVertices const& expressions)
{
  assert(false);
  return UnicodeString();

  // UnicodeString result;

  // std::vector<UnicodeString> scripts;
  // BOOST_FOREACH(boost::shared_ptr<ranally::ExpressionVertex> expressionVertex,
  //   expressions) {
  //   // result += expressionVertex->Accept(*this);

  //   result += indent(expressionVertex->Accept(*this));
  //   result += ";\n";
  // }

  // return result;
}



UnicodeString DotVisitor::Visit(
  AssignmentVertex& vertex)
{
  ExpressionVertices const& targets = vertex.targets();
  assert(targets.size() == 1);

  ExpressionVertices const& expressions = vertex.expressions();
  assert(expressions.size() == 1);

  UnicodeString result;
  result += indent(expressions[0]->name());
  result += " -> ";
  result += targets[0]->name();
  result += ";\n";

  // BOOST_FOREACH(boost::shared_ptr<ranally::ExpressionVertex> expressionVertex,
  //   vertex.expressions()) {
  //   result += expressionVertex->Accept(*this);
  // }

  return result;
}



UnicodeString DotVisitor::Visit(
  FunctionVertex& vertex)
{
  UnicodeString result;

  if(vertex.expressions().empty()) {
    result += indent(vertex.name());
    result += ";\n";
  }
  else {
    BOOST_FOREACH(boost::shared_ptr<ranally::ExpressionVertex> expressionVertex,
      vertex.expressions()) {
      result += indent(expressionVertex->name());
      result += " -> ";
      result += vertex.name();
      result += ";\n";
    }
  }

  return result;
}



UnicodeString DotVisitor::Visit(
  OperatorVertex& vertex)
{
  // assert(vertex.expressions().size() == 1 || vertex.expressions().size() == 2);
  // UnicodeString result;

  // if(vertex.expressions().size() == 1) {
  //   // Unary operator.
  //   if(vertex.name() == "Invert") {
  //     result += "~";
  //   }
  //   else if(vertex.name() == "Not") {
  //     result += "!";
  //   }
  //   else if(vertex.name() == "Add") {
  //     result += "+";
  //   }
  //   else if(vertex.name() == "Sub") {
  //     result += "-";
  //   }
  //   else {
  //     // TODO
  //     assert(false);
  //   }

  //   result += "(" + vertex.expressions()[0]->Accept(*this) + ")";
  // }
  // else if(vertex.expressions().size() == 2) {
  //   // Binary operator.
  //   result += "(" + vertex.expressions()[0]->Accept(*this) + ") ";

  //   if(vertex.name() == "Add") {
  //     result += "+";
  //   }
  //   else if(vertex.name() == "Sub") {
  //     result += "-";
  //   }
  //   else if(vertex.name() == "Mult") {
  //     result += "*";
  //   }
  //   else if(vertex.name() == "Div") {
  //     result += "/";
  //   }
  //   else if(vertex.name() == "Mod") {
  //     result += "%";
  //   }
  //   else if(vertex.name() == "Pow") {
  //     result += "**";
  //   }
  //   else if(vertex.name() == "LShift") {
  //     // TODO
  //     assert(false);
  //     result += "";
  //   }
  //   else if(vertex.name() == "RShift") {
  //     // TODO
  //     assert(false);
  //     result += "";
  //   }
  //   else if(vertex.name() == "BitOr") {
  //     // TODO
  //     assert(false);
  //     result += "";
  //   }
  //   else if(vertex.name() == "BitXor") {
  //     // TODO
  //     assert(false);
  //     result += "";
  //   }
  //   else if(vertex.name() == "BitAnd") {
  //     // TODO
  //     assert(false);
  //     result += "";
  //   }
  //   else if(vertex.name() == "FloorDiv") {
  //     // TODO
  //     assert(false);
  //     result += "";
  //   }
  //   else {
  //     // TODO
  //     assert(false);
  //   }

  //   result += " (" + vertex.expressions()[1]->Accept(*this) + ")";
  // }

  // return result;
  return UnicodeString();
}



UnicodeString DotVisitor::Visit(
  SyntaxVertex&)
{
  assert(false);
  return UnicodeString();
}



UnicodeString DotVisitor::Visit(
  ScriptVertex& vertex)
{
  _indentLevel = 0;
  UnicodeString result = "digraph G {\n";
  ++_indentLevel;
  result += visitStatements(vertex.statements());
  --_indentLevel;
  result += "}\n";
  assert(_indentLevel == 0);
  return result; // TODO result.replace("\"", "\\\"");
}



UnicodeString DotVisitor::Visit(
  StringVertex& vertex)
{
  return indent("\"" + vertex.value() + "\";\n");
}



UnicodeString DotVisitor::Visit(
  NameVertex& vertex)
{
  return indent(vertex.name()) + ";\n";
}



template<typename T>
UnicodeString DotVisitor::Visit(
  NumberVertex<T>& vertex)
{
  return indent(UnicodeString((boost::format("%1%;\n") % vertex.value()).str().c_str()));
}



UnicodeString DotVisitor::Visit(
  NumberVertex<int8_t>& vertex)
{
  return Visit<int8_t>(vertex);
}



UnicodeString DotVisitor::Visit(
  NumberVertex<int16_t>& vertex)
{
  return Visit<int16_t>(vertex);
}



UnicodeString DotVisitor::Visit(
  NumberVertex<int32_t>& vertex)
{
  return Visit<int32_t>(vertex);
}



UnicodeString DotVisitor::Visit(
  NumberVertex<int64_t>& vertex)
{
  return Visit<int64_t>(vertex);
}



UnicodeString DotVisitor::Visit(
  NumberVertex<uint8_t>& vertex)
{
  return Visit<uint8_t>(vertex);
}



UnicodeString DotVisitor::Visit(
  NumberVertex<uint16_t>& vertex)
{
  return Visit<uint16_t>(vertex);
}



UnicodeString DotVisitor::Visit(
  NumberVertex<uint32_t>& vertex)
{
  return Visit<uint32_t>(vertex);
}



UnicodeString DotVisitor::Visit(
  NumberVertex<uint64_t>& vertex)
{
  return Visit<uint64_t>(vertex);
}



UnicodeString DotVisitor::Visit(
  NumberVertex<float>& vertex)
{
  return Visit<float>(vertex);
}



UnicodeString DotVisitor::Visit(
  NumberVertex<double>& vertex)
{
  return Visit<double>(vertex);
}



UnicodeString DotVisitor::Visit(
  IfVertex& vertex)
{
  // assert(!vertex.trueStatements().empty());

  // UnicodeString result;

  // // The indent function called in visitStatements of the parent vertex
  // // indents the first line of this if-statement, so we have to indent the
  // // else line ourselves.
  // // The statements that are part of the true and false blocks are indented
  // // by the visitStatements.
  // result += "if " + vertex.condition()->Accept(*this) + ":\n";
  // ++_indentLevel;
  // result += visitStatements(vertex.trueStatements());
  // --_indentLevel;

  // if(!vertex.falseStatements().empty()) {
  //   result += indent("else:\n");
  //   ++_indentLevel;
  //   result += visitStatements(vertex.falseStatements());
  //   --_indentLevel;
  // }

  // return result;
  return UnicodeString();
}



UnicodeString DotVisitor::Visit(
  WhileVertex& vertex)
{
  // assert(!vertex.trueStatements().empty());

  // UnicodeString result;

  // // The indent function called in visitStatements of the parent vertex
  // // indents the first line of this while-statement, so we have to indent the
  // // else line ourselves.
  // // The statements that are part of the true and false blocks are indented
  // // by the visitStatements.
  // result += "while " + vertex.condition()->Accept(*this) + ":\n";
  // ++_indentLevel;
  // result += visitStatements(vertex.trueStatements());
  // --_indentLevel;

  // if(!vertex.falseStatements().empty()) {
  //   result += indent("else:\n");
  //   ++_indentLevel;
  //   result += visitStatements(vertex.falseStatements());
  //   --_indentLevel;
  // }

  // return result;
  return UnicodeString();
}

} // namespace ranally

