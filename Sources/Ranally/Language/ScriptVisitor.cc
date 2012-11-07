#include "Ranally/Language/ScriptVisitor.h"
#include "Ranally/Language/Vertices.h"


namespace ranally {

ScriptVisitor::ScriptVisitor(
    size_t tabSize)

    : Visitor(),
      _tabSize(tabSize),
      _indentLevel(0)

{
}


String const& ScriptVisitor::script() const
{
    return _script;
}


// String ScriptVisitor::indent(
//   String const& statement)
// {
//   // Only the first line of multi-line statements (if-statement) is indented
//   // here.
//   String indentation = std::string(_indentLevel * _tabSize, ' ').c_str();
//   return indentation + statement;
// }


String ScriptVisitor::indentation() const
{
    return String(std::string(_indentLevel * _tabSize, ' '));
}


void ScriptVisitor::visitStatements(
    StatementVertices& statements)
{
    for(auto statementVertex: statements) {
        _script += indentation();
        statementVertex->Accept(*this);

        if(!_script.endsWith("\n")) {
            _script += "\n";
        }
    }
}


void ScriptVisitor::visitExpressions(
    ExpressionVertices const& expressions)
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
    AssignmentVertex& vertex)
{
    vertex.target()->Accept(*this);
    _script += " = ";
    vertex.expression()->Accept(*this);
}


void ScriptVisitor::Visit(
    FunctionVertex& vertex)
{
    _script += vertex.name();
    visitExpressions(vertex.expressions());
}


void ScriptVisitor::Visit(
    OperatorVertex& vertex)
{
    assert(vertex.expressions().size() == 1 ||
        vertex.expressions().size() == 2);

    if(vertex.expressions().size() == 1) {
        // Unary operator.
        _script += vertex.symbol();
        _script += "(";
        vertex.expressions()[0]->Accept(*this);
        _script += ")";
    }
    else if(vertex.expressions().size() == 2) {
        // Binary operator.
        _script += "(";
        vertex.expressions()[0]->Accept(*this);
        _script += ") ";

        _script += vertex.symbol();

        _script += " (";
        vertex.expressions()[1]->Accept(*this);
        _script += ")";
    }
}


void ScriptVisitor::Visit(
    SyntaxVertex&)
{
    assert(false);
}


void ScriptVisitor::Visit(
    ScriptVertex& vertex)
{
    _indentLevel = 0;
    _script = String();
    visitStatements(vertex.statements());
    assert(_indentLevel == 0);
}


void ScriptVisitor::Visit(
    StringVertex& vertex)
{
    _script += "\"" + vertex.value() + "\"";
}


void ScriptVisitor::Visit(
    NameVertex& vertex)
{
    _script += vertex.name();
}


void ScriptVisitor::Visit(
    NumberVertex<int8_t>& vertex)
{
    _script += String(boost::format("%1%") % vertex.value());
}


void ScriptVisitor::Visit(
    NumberVertex<int16_t>& vertex)
{
    _script += String(boost::format("%1%") % vertex.value());
}


void ScriptVisitor::Visit(
    NumberVertex<int32_t>& vertex)
{
    _script += String(boost::format("%1%") % vertex.value());
}


void ScriptVisitor::Visit(
    NumberVertex<int64_t>& vertex)
{
    std::string formatString = sizeof(long) == sizeof(int64_t) ? "%1%" : "%1%L";
    _script += String(boost::format(formatString) % vertex.value());
}


void ScriptVisitor::Visit(
    NumberVertex<uint8_t>& vertex)
{
    // U?
    _script += String(boost::format("%1%U") % vertex.value());
}


void ScriptVisitor::Visit(
    NumberVertex<uint16_t>& vertex)
{
    // U?
    _script += String(boost::format("%1%U") % vertex.value());
}


void ScriptVisitor::Visit(
    NumberVertex<uint32_t>& vertex)
{
    // U?
    _script += String(boost::format("%1%U") % vertex.value());
}


void ScriptVisitor::Visit(
    NumberVertex<uint64_t>& vertex)
{
    // U?
    std::string formatString = sizeof(unsigned long) == sizeof(uint64_t)
      ? "%1%U" : "%1%UL";
    _script += String(boost::format(formatString) % vertex.value());
}


void ScriptVisitor::Visit(
    NumberVertex<float>& vertex)
{
    _script += String(boost::format("%1%") % vertex.value());
}


void ScriptVisitor::Visit(
    NumberVertex<double>& vertex)
{
    _script += String(boost::format("%1%") % vertex.value());
}


void ScriptVisitor::Visit(
    IfVertex& vertex)
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
    WhileVertex& vertex)
{
    assert(!vertex.trueStatements().empty());

    String result;

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
