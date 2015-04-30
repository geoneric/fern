// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/ast/visitor/module_visitor.h"
#include "fern/language/ast/core/vertices.h"


namespace fern {

ModuleVisitor::ModuleVisitor(
    size_t tab_size)

    : AstVisitor(),
      _tab_size(tab_size),
      _indent_level(0)

{
}


String const& ModuleVisitor::module() const
{
    return _module;
}


// String ModuleVisitor::indent(
//   String const& statement)
// {
//   // Only the first line of multi-line statements (if-statement) is indented
//   // here.
//   String indentation = std::string(_indent_level * _tab_size, ' ').c_str();
//   return indentation + statement;
// }


String ModuleVisitor::indentation() const
{
    return String(std::string(_indent_level * _tab_size, ' '));
}


void ModuleVisitor::visit_statements(
    StatementVertices& statements)
{
    for(auto statement_vertex: statements) {
        _module += indentation();
        statement_vertex->Accept(*this);

        if(!_module.ends_with("\n")) {
            _module += "\n";
        }
    }
}


void ModuleVisitor::visit_expressions(
    ExpressionVertices const& expressions)
{
    _module += "(";

    for(size_t i = 0; i < expressions.size(); ++i) {
        expressions[i]->Accept(*this);

        if(i < expressions.size() - 1) {
            _module += ", ";
        }
    }

    _module += ")";
}


void ModuleVisitor::Visit(
    AssignmentVertex& vertex)
{
    vertex.target()->Accept(*this);
    _module += " = ";
    vertex.expression()->Accept(*this);
}


void ModuleVisitor::Visit(
    FunctionCallVertex& vertex)
{
    _module += vertex.name();
    visit_expressions(vertex.expressions());
}


void ModuleVisitor::Visit(
    OperatorVertex& vertex)
{
    assert(vertex.expressions().size() == 1 ||
        vertex.expressions().size() == 2);

    if(vertex.expressions().size() == 1) {
        // Unary operator.
        _module += vertex.symbol();
        _module += "(";
        vertex.expressions()[0]->Accept(*this);
        _module += ")";
    }
    else if(vertex.expressions().size() == 2) {
        // Binary operator.
        _module += "(";
        vertex.expressions()[0]->Accept(*this);
        _module += ") ";

        _module += vertex.symbol();

        _module += " (";
        vertex.expressions()[1]->Accept(*this);
        _module += ")";
    }
}


void ModuleVisitor::Visit(
    AstVertex&)
{
    assert(false);
}


void ModuleVisitor::Visit(
    ModuleVertex& vertex)
{
    _indent_level = 0;
    _module = String();
    visit_statements(vertex.scope()->statements());
    assert(_indent_level == 0);
}


void ModuleVisitor::Visit(
    StringVertex& vertex)
{
    _module += String("\"") + vertex.value() + String("\"");
}


void ModuleVisitor::Visit(
    NameVertex& vertex)
{
    _module += vertex.name();
}


void ModuleVisitor::Visit(
    SubscriptVertex& vertex)
{
    _module += "(";
    vertex.expression()->Accept(*this);
    _module += ")";
    _module += "[";
    vertex.selection()->Accept(*this);
    _module += "]";
}


void ModuleVisitor::Visit(
    AttributeVertex& vertex)
{
    _module += "(";
    vertex.expression()->Accept(*this);
    _module += ")";
    _module += ".";
    _module += vertex.member_name();
}


void ModuleVisitor::Visit(
    NumberVertex<int8_t>& vertex)
{
    _module += String(boost::format("%1%") % vertex.value());
}


void ModuleVisitor::Visit(
    NumberVertex<int16_t>& vertex)
{
    _module += String(boost::format("%1%") % vertex.value());
}


void ModuleVisitor::Visit(
    NumberVertex<int32_t>& vertex)
{
    _module += String(boost::format("%1%") % vertex.value());
}


void ModuleVisitor::Visit(
    NumberVertex<int64_t>& vertex)
{
    std::string format_string = sizeof(long) == sizeof(int64_t)
        ? "%1%" : "%1%L";
    _module += String(boost::format(format_string) % vertex.value());
}


void ModuleVisitor::Visit(
    NumberVertex<uint8_t>& vertex)
{
    // U?
    _module += String(boost::format("%1%U") % vertex.value());
}


void ModuleVisitor::Visit(
    NumberVertex<uint16_t>& vertex)
{
    // U?
    _module += String(boost::format("%1%U") % vertex.value());
}


void ModuleVisitor::Visit(
    NumberVertex<uint32_t>& vertex)
{
    // U?
    _module += String(boost::format("%1%U") % vertex.value());
}


void ModuleVisitor::Visit(
    NumberVertex<uint64_t>& vertex)
{
    // U?
    std::string format_string = sizeof(unsigned long) == sizeof(uint64_t)
      ? "%1%U" : "%1%UL";
    _module += String(boost::format(format_string) % vertex.value());
}


void ModuleVisitor::Visit(
    NumberVertex<float>& vertex)
{
    _module += String(boost::format("%1%") % vertex.value());
}


void ModuleVisitor::Visit(
    NumberVertex<double>& vertex)
{
    _module += String(boost::format("%1%") % vertex.value());
}


void ModuleVisitor::Visit(
    IfVertex& vertex)
{
    // The indent function called in visit_statements of the parent vertex
    // indents the first line of this if-statement, so we have to indent the
    // else line ourselves.
    // The statements that are part of the true and false blocks are indented
    // by the visit_statements.
    _module += "if ";
    vertex.condition()->Accept(*this);
    _module += ":\n";

    ++_indent_level;
    visit_statements(vertex.true_scope()->statements());
    --_indent_level;

    if(!vertex.false_scope()->statements().empty()) {
        _module += indentation();
        _module += "else:\n";
        ++_indent_level;
        visit_statements(vertex.false_scope()->statements());
        --_indent_level;
    }
}


void ModuleVisitor::Visit(
    WhileVertex& vertex)
{
    String result;

    // The indent function called in visit_statements of the parent vertex
    // indents the first line of this while-statement, so we have to indent the
    // else line ourselves.
    // The statements that are part of the true and false blocks are indented
    // by the visit_statements.
    _module += "while ";
    vertex.condition()->Accept(*this);
    _module += ":\n";
    ++_indent_level;
    visit_statements(vertex.true_scope()->statements());
    --_indent_level;

    if(!vertex.false_scope()->statements().empty()) {
        _module += indentation();
        _module += "else:\n";
        ++_indent_level;
        visit_statements(vertex.false_scope()->statements());
        --_indent_level;
    }
}

} // namespace fern
