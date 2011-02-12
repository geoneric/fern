#include "Ranally/Language/ScriptVisitor.h"

#include <boost/foreach.hpp>
#include <boost/format.hpp>
#include "Ranally/Language/Vertices.h"



namespace ranally {

ScriptVisitor::ScriptVisitor(
  size_t tabSize)

  : _tabSize(tabSize),
    _indentLevel(0)

{
}



ScriptVisitor::~ScriptVisitor()
{
}



UnicodeString const& ScriptVisitor::script() const
{
  return _script;
}



// UnicodeString ScriptVisitor::indent(
//   UnicodeString const& statement)
// {
//   // Only the first line of multi-line statements (if-statement) is indented
//   // here.
//   UnicodeString indentation = std::string(_indentLevel * _tabSize, ' ').c_str();
//   return indentation + statement;
// }



UnicodeString ScriptVisitor::indentation() const
{
  return UnicodeString(std::string(_indentLevel * _tabSize, ' ').c_str());
}



void ScriptVisitor::visitStatements(
  language::StatementVertices const& statements)
{
  BOOST_FOREACH(boost::shared_ptr<language::StatementVertex> statementVertex,
    statements) {
    _script += indentation();
    statementVertex->Accept(*this);

    if(!_script.endsWith("\n")) {
      _script += "\n";
    }
  }
}



void ScriptVisitor::visitExpressions(
  language::ExpressionVertices const& expressions)
{
  _script += "(";

  for(size_t i = 0; i < expressions.size(); ++i) {
    expressions[i]->Accept(*this);

    if(i < expressions.size() - 1) {
      _script += ", ";
    }
  }

  _script += ")";
}



void ScriptVisitor::Visit(
  language::AssignmentVertex& vertex)
{
  language::ExpressionVertices const& targets = vertex.targets();
  assert(targets.size() == 1);

  language::ExpressionVertices const& expressions = vertex.expressions();
  assert(expressions.size() == 1);

  targets[0]->Accept(*this);
  _script += " = ";
  expressions[0]->Accept(*this);
}



void ScriptVisitor::Visit(
  language::FunctionVertex& vertex)
{
  _script += vertex.name();
  visitExpressions(vertex.expressions());
}



void ScriptVisitor::Visit(
  language::OperatorVertex& vertex)
{
  assert(vertex.expressions().size() == 1 || vertex.expressions().size() == 2);

  if(vertex.expressions().size() == 1) {
    // Unary operator.
    if(vertex.name() == "Invert") {
      _script += "~";
    }
    else if(vertex.name() == "Not") {
      _script += "!";
    }
    else if(vertex.name() == "Add") {
      _script += "+";
    }
    else if(vertex.name() == "Sub") {
      _script += "-";
    }
    else {
      // TODO
      assert(false);
    }

    _script += "(";
    vertex.expressions()[0]->Accept(*this);
    _script += ")";
  }
  else if(vertex.expressions().size() == 2) {
    // Binary operator.
    _script += "(";
    vertex.expressions()[0]->Accept(*this);
    _script += ") ";

    if(vertex.name() == "Add") {
      _script += "+";
    }
    else if(vertex.name() == "Sub") {
      _script += "-";
    }
    else if(vertex.name() == "Mult") {
      _script += "*";
    }
    else if(vertex.name() == "Div") {
      _script += "/";
    }
    else if(vertex.name() == "Mod") {
      _script += "%";
    }
    else if(vertex.name() == "Pow") {
      _script += "**";
    }
    else if(vertex.name() == "LShift") {
      // TODO
      assert(false);
      _script += "";
    }
    else if(vertex.name() == "RShift") {
      // TODO
      assert(false);
      _script += "";
    }
    else if(vertex.name() == "BitOr") {
      // TODO
      assert(false);
      _script += "";
    }
    else if(vertex.name() == "BitXor") {
      // TODO
      assert(false);
      _script += "";
    }
    else if(vertex.name() == "BitAnd") {
      // TODO
      assert(false);
      _script += "";
    }
    else if(vertex.name() == "FloorDiv") {
      // TODO
      assert(false);
      _script += "";
    }
    else {
      // TODO
      assert(false);
    }

    _script += " (";
    vertex.expressions()[1]->Accept(*this);
    _script += ")";
  }
}



void ScriptVisitor::Visit(
  language::SyntaxVertex&)
{
  assert(false);
}



void ScriptVisitor::Visit(
  language::ScriptVertex& vertex)
{
  _indentLevel = 0;
  _script = UnicodeString();
  visitStatements(vertex.statements());
  assert(_indentLevel == 0);
}



void ScriptVisitor::Visit(
  language::StringVertex& vertex)
{
  _script += "\"" + vertex.value() + "\"";
}



void ScriptVisitor::Visit(
  language::NameVertex& vertex)
{
  _script += vertex.name();
}



void ScriptVisitor::Visit(
  language::NumberVertex<int8_t>& vertex)
{
  _script += UnicodeString((boost::format("%1%") % vertex.value()).str().c_str());
}



void ScriptVisitor::Visit(
  language::NumberVertex<int16_t>& vertex)
{
  _script += UnicodeString((boost::format("%1%") % vertex.value()).str().c_str());
}



void ScriptVisitor::Visit(
  language::NumberVertex<int32_t>& vertex)
{
  _script += UnicodeString((boost::format("%1%") % vertex.value()).str().c_str());
}



void ScriptVisitor::Visit(
  language::NumberVertex<int64_t>& vertex)
{
  _script += UnicodeString((boost::format("%1%L") % vertex.value()).str().c_str());
}



void ScriptVisitor::Visit(
  language::NumberVertex<uint8_t>& vertex)
{
  // U?
  _script += UnicodeString((boost::format("%1%U") % vertex.value()).str().c_str());
}



void ScriptVisitor::Visit(
  language::NumberVertex<uint16_t>& vertex)
{
  // U?
  _script += UnicodeString((boost::format("%1%U") % vertex.value()).str().c_str());
}



void ScriptVisitor::Visit(
  language::NumberVertex<uint32_t>& vertex)
{
  // U?
  _script += UnicodeString((boost::format("%1%U") % vertex.value()).str().c_str());
}



void ScriptVisitor::Visit(
  language::NumberVertex<uint64_t>& vertex)
{
  // U?
  _script += UnicodeString((boost::format("%1%UL") % vertex.value()).str().c_str());
}



void ScriptVisitor::Visit(
  language::NumberVertex<float>& vertex)
{
  _script += UnicodeString((boost::format("%1%") % vertex.value()).str().c_str());
}



void ScriptVisitor::Visit(
  language::NumberVertex<double>& vertex)
{
  _script += UnicodeString((boost::format("%1%") % vertex.value()).str().c_str());
}



void ScriptVisitor::Visit(
  language::IfVertex& vertex)
{
  assert(!vertex.trueStatements().empty());

  // The indent function called in visitStatements of the parent vertex
  // indents the first line of this if-statement, so we have to indent the
  // else line ourselves.
  // The statements that are part of the true and false blocks are indented
  // by the visitStatements.
  _script += "if ";
  vertex.condition()->Accept(*this);
  _script += ":\n";

  ++_indentLevel;
  visitStatements(vertex.trueStatements());
  --_indentLevel;

  if(!vertex.falseStatements().empty()) {
    _script += indentation();
    _script += "else:\n";
    ++_indentLevel;
    visitStatements(vertex.falseStatements());
    --_indentLevel;
  }
}



void ScriptVisitor::Visit(
  language::WhileVertex& vertex)
{
  assert(!vertex.trueStatements().empty());

  UnicodeString result;

  // The indent function called in visitStatements of the parent vertex
  // indents the first line of this while-statement, so we have to indent the
  // else line ourselves.
  // The statements that are part of the true and false blocks are indented
  // by the visitStatements.
  _script += "while ";
  vertex.condition()->Accept(*this);
  _script += ":\n";
  ++_indentLevel;
  visitStatements(vertex.trueStatements());
  --_indentLevel;

  if(!vertex.falseStatements().empty()) {
    _script += indentation();
    _script += "else:\n";
    ++_indentLevel;
    visitStatements(vertex.falseStatements());
    --_indentLevel;
  }
}

} // namespace ranally

