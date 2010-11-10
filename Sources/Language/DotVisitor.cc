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
{
}



DotVisitor::~DotVisitor()
{
}



UnicodeString DotVisitor::indent(
  UnicodeString const& statement)
{
  // // Only the first line of multi-line statements (if-statement) is indented
  // // here.
  // UnicodeString indentation = std::string(_indentLevel * _tabSize, ' ').c_str();
  // return indentation + statement;
  return UnicodeString();
}



UnicodeString DotVisitor::visitStatements(
  StatementVertices const& statements)
{
  // UnicodeString result;

  // BOOST_FOREACH(boost::shared_ptr<ranally::StatementVertex> statementVertex,
  //   statements) {
  //   result += indent(statementVertex->Accept(*this));

  //   if(!result.endsWith("\n")) {
  //     result += "\n";
  //   }
  // }

  // return result;
  return UnicodeString();
}



UnicodeString DotVisitor::visitExpressions(
  ExpressionVertices const& expressions)
{
  // UnicodeString result = "(";

  // std::vector<UnicodeString> scripts;
  // BOOST_FOREACH(boost::shared_ptr<ranally::ExpressionVertex> expressionVertex,
  //   expressions) {
  //   scripts.push_back(expressionVertex->Accept(*this));
  // }

  // if(!scripts.empty()) {
  //   result += scripts[0];

  //   for(size_t i = 1; i < scripts.size(); ++i) {
  //     result += ", " + scripts[i];
  //   }
  // }

  // result += ")";

  // return result;
  return UnicodeString();
}



UnicodeString DotVisitor::Visit(
  AssignmentVertex& vertex)
{
  // ExpressionVertices const& targets = vertex.targets();
  // assert(targets.size() == 1);

  // ExpressionVertices const& expressions = vertex.expressions();
  // assert(expressions.size() == 1);

  // UnicodeString result;
  // result += targets[0]->Accept(*this);
  // result += " = ";
  // result += expressions[0]->Accept(*this);

  // return result;
  return UnicodeString();
}



UnicodeString DotVisitor::Visit(
  FunctionVertex& vertex)
{
  // UnicodeString result = vertex.name();
  // result += visitExpressions(vertex.expressions());
  // return result;
  return UnicodeString();
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
  // _indentLevel = 0;
  // UnicodeString result = visitStatements(vertex.statements());
  // assert(_indentLevel == 0);
  // return result;
  return UnicodeString();
}



UnicodeString DotVisitor::Visit(
  StringVertex& vertex)
{
  // return "\"" + vertex.value() + "\"";
  return UnicodeString();
}



UnicodeString DotVisitor::Visit(
  NameVertex& vertex)
{
  // return vertex.name();
  return UnicodeString();
}



UnicodeString DotVisitor::Visit(
  NumberVertex<int8_t>& vertex)
{
  // return UnicodeString((boost::format("%1%") % vertex.value()).str().c_str());
  return UnicodeString();
}



UnicodeString DotVisitor::Visit(
  NumberVertex<int16_t>& vertex)
{
  // return UnicodeString((boost::format("%1%") % vertex.value()).str().c_str());
  return UnicodeString();
}



UnicodeString DotVisitor::Visit(
  NumberVertex<int32_t>& vertex)
{
  // return UnicodeString((boost::format("%1%") % vertex.value()).str().c_str());
  return UnicodeString();
}



UnicodeString DotVisitor::Visit(
  NumberVertex<int64_t>& vertex)
{
  // return UnicodeString((boost::format("%1%L") % vertex.value()).str().c_str());
  return UnicodeString();
}



UnicodeString DotVisitor::Visit(
  NumberVertex<uint8_t>& vertex)
{
  // U?
  // return UnicodeString((boost::format("%1%U") % vertex.value()).str().c_str());
  return UnicodeString();
}



UnicodeString DotVisitor::Visit(
  NumberVertex<uint16_t>& vertex)
{
  // U?
  // return UnicodeString((boost::format("%1%U") % vertex.value()).str().c_str());
  return UnicodeString();
}



UnicodeString DotVisitor::Visit(
  NumberVertex<uint32_t>& vertex)
{
  // U?
  // return UnicodeString((boost::format("%1%U") % vertex.value()).str().c_str());
  return UnicodeString();
}



UnicodeString DotVisitor::Visit(
  NumberVertex<uint64_t>& vertex)
{
  // U?
  // return UnicodeString((boost::format("%1%UL") % vertex.value()).str().c_str());
  return UnicodeString();
}



UnicodeString DotVisitor::Visit(
  NumberVertex<float>& vertex)
{
  // return UnicodeString((boost::format("%1%") % vertex.value()).str().c_str());
  return UnicodeString();
}



UnicodeString DotVisitor::Visit(
  NumberVertex<double>& vertex)
{
  // return UnicodeString((boost::format("%1%") % vertex.value()).str().c_str());
  return UnicodeString();
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

