#include "ranally/language/script_visitor.h"
#include "ranally/language/vertices.h"


namespace ranally {

ScriptVisitor::ScriptVisitor(
    size_t tab_size)

    : Visitor(),
      _tab_size(tab_size),
      _indent_level(0)

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
//   String indentation = std::string(_indent_level * _tab_size, ' ').c_str();
//   return indentation + statement;
// }


String ScriptVisitor::indentation() const
{
    return String(std::string(_indent_level * _tab_size, ' '));
}


void ScriptVisitor::visit_statements(
    StatementVertices& statements)
{
    for(auto statement_vertex: statements) {
        _script += indentation();
        statement_vertex->Accept(*this);

        if(!_script.ends_with("\n")) {
            _script += "\n";
        }
    }
}


void ScriptVisitor::visit_expressions(
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
    visit_expressions(vertex.expressions());
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
    _indent_level = 0;
    _script = String();
    visit_statements(vertex.statements());
    assert(_indent_level == 0);
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
    SubscriptVertex& vertex)
{
    _script += "(";
    vertex.expression()->Accept(*this);
    _script += ")";
    _script += "[";
    vertex.selection()->Accept(*this);
    _script += "]";
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
    std::string format_string = sizeof(long) == sizeof(int64_t)
        ? "%1%" : "%1%L";
    _script += String(boost::format(format_string) % vertex.value());
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
    std::string format_string = sizeof(unsigned long) == sizeof(uint64_t)
      ? "%1%U" : "%1%UL";
    _script += String(boost::format(format_string) % vertex.value());
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
    assert(!vertex.true_statements().empty());

    // The indent function called in visit_statements of the parent vertex
    // indents the first line of this if-statement, so we have to indent the
    // else line ourselves.
    // The statements that are part of the true and false blocks are indented
    // by the visit_statements.
    _script += "if ";
    vertex.condition()->Accept(*this);
    _script += ":\n";

    ++_indent_level;
    visit_statements(vertex.true_statements());
    --_indent_level;

    if(!vertex.false_statements().empty()) {
        _script += indentation();
        _script += "else:\n";
        ++_indent_level;
        visit_statements(vertex.false_statements());
        --_indent_level;
    }
}


void ScriptVisitor::Visit(
    WhileVertex& vertex)
{
    assert(!vertex.true_statements().empty());

    String result;

    // The indent function called in visit_statements of the parent vertex
    // indents the first line of this while-statement, so we have to indent the
    // else line ourselves.
    // The statements that are part of the true and false blocks are indented
    // by the visit_statements.
    _script += "while ";
    vertex.condition()->Accept(*this);
    _script += ":\n";
    ++_indent_level;
    visit_statements(vertex.true_statements());
    --_indent_level;

    if(!vertex.false_statements().empty()) {
        _script += indentation();
        _script += "else:\n";
        ++_indent_level;
        visit_statements(vertex.false_statements());
        --_indent_level;
    }
}

} // namespace ranally
