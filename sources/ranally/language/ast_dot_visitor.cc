#include "ranally/language/ast_dot_visitor.h"
#include "ranally/core/string.h"
#include "ranally/language/vertices.h"


namespace ranally {
namespace {

String join(
    std::vector<String> const& strings,
    String const& separator)
{
    String result;

    if(!strings.empty()) {
        result += strings.front();

        for(size_t i = 1; i < strings.size(); ++i) {
            result += separator + strings[i];
        }
    }

    return result;
}


String data_types_to_string(
    DataTypes const& data_types)
{
    std::vector<String> strings;

    if(data_types & DataType::DT_UNKNOWN) {
        strings.push_back("?");
    }
    else {
        if(data_types & DataType::DT_VALUE) {
            strings.push_back("val");
        }
        if(data_types & DataType::DT_RASTER) {
            strings.push_back("rst");
        }
        if(data_types & DataType::DT_FEATURE) {
            strings.push_back("ftr");
        }
        if(data_types & DataType::DT_DEPENDS_ON_INPUT) {
            strings.push_back("dep");
        }
    }

    assert(!strings.empty());
    return join(strings, "|");
}


String value_types_to_string(
    ValueTypes const& value_types)
{
    std::vector<String> strings;

    if(value_types & DataType::DT_UNKNOWN) {
        strings.push_back("?");
    }
    else {
        if(value_types & VT_UINT8) {
            strings.push_back("u8");
        }
        if(value_types & VT_INT8) {
            strings.push_back("s8");
        }
        if(value_types & VT_UINT16) {
            strings.push_back("u16");
        }
        if(value_types & VT_INT16) {
            strings.push_back("s16");
        }
        if(value_types & VT_UINT32) {
            strings.push_back("u32");
        }
        if(value_types & VT_INT32) {
            strings.push_back("s32");
        }
        if(value_types & VT_UINT64) {
            strings.push_back("u64");
        }
        if(value_types & VT_INT64) {
            strings.push_back("s64");
        }
        if(value_types & VT_FLOAT32) {
            strings.push_back("f32");
        }
        if(value_types & VT_FLOAT64) {
            strings.push_back("f64");
        }
        if(value_types & VT_STRING) {
            strings.push_back("str");
        }
        if(value_types & VT_DEPENDS_ON_INPUT) {
            strings.push_back("dep");
        }
    }

    assert(!strings.empty());
    return join(strings, "|");
}

} // Anonymous namespace


AstDotVisitor::AstDotVisitor(
    int modes)

    : DotVisitor(),
      _mode(Mode::Declaring),
      _modes(modes)

{
}


void AstDotVisitor::set_mode(
    Mode mode)
{
    _mode = mode;
}


void AstDotVisitor::add_ast_vertex(
    SyntaxVertex const& source_vertex,
    SyntaxVertex const& target_vertex)
{
    assert(_mode == Mode::ConnectingAst);
    add_script(
        String(boost::format("\"%1%\"") % &source_vertex) + " -> " +
        String(boost::format("\"%1%\"") % &target_vertex) + " ["
        "];\n"
    );
}


void AstDotVisitor::add_cfg_vertices(
    SyntaxVertex const& source_vertex)
{
    assert(_mode == Mode::ConnectingCfg);
    for(auto successor: source_vertex.successors()) {
        add_script(
            String(boost::format("\"%1%\"") % &source_vertex) + " -> " +
            String(boost::format("\"%1%\"") % successor) + " ["
                "color=\"/spectral9/2\", "
                "constraint=false, "
                "style=dashed, "
                "penwidth=0.25"
            "];\n"
        );
    }
}


void AstDotVisitor::add_use_vertices(
    NameVertex const& vertex)
{
    assert(_mode == Mode::ConnectingUses);
    for(auto use: vertex.uses()) {
        add_script(
            String(boost::format("\"%1%\"") % &vertex) + " -> " +
            String(boost::format("\"%1%\"") % use) + " ["
              "color=\"/spectral9/8\", "
              "constraint=false, "
              "style=dashed, "
              "penwidth=0.25"
            "];\n"
        );
    }
}


template<typename T>
void AstDotVisitor::Visit(
    NumberVertex<T>& vertex)
{
    switch(_mode) {
        case Mode::Declaring: {
            add_script(
                String(boost::format("\"%1%\"") % &vertex) +
                " [label=\"" + String(boost::format("%1%") % vertex.value()) +
                "\", fontname=courier, shape=box];\n"
            );
            break;
        }
        case Mode::ConnectingAst: {
            break;
        }
        case Mode::ConnectingCfg: {
            add_cfg_vertices(vertex);
            break;
        }
        case Mode::ConnectingUses: {
            break;
        }
    }
}


#define VISIT_NUMBER_VERTEX(                                                   \
    type)                                                                      \
void AstDotVisitor::Visit(                                                     \
    NumberVertex<type>& vertex)                                      \
{                                                                              \
    Visit<type>(vertex);                                                       \
}

VISIT_NUMBER_VERTEX(int8_t  )
VISIT_NUMBER_VERTEX(int16_t )
VISIT_NUMBER_VERTEX(int32_t )
VISIT_NUMBER_VERTEX(int64_t )
VISIT_NUMBER_VERTEX(uint8_t )
VISIT_NUMBER_VERTEX(uint16_t)
VISIT_NUMBER_VERTEX(uint32_t)
VISIT_NUMBER_VERTEX(uint64_t)
VISIT_NUMBER_VERTEX(float   )
VISIT_NUMBER_VERTEX(double  )

#undef VISIT_NUMBER_VERTEX


void AstDotVisitor::Visit(
    AssignmentVertex& vertex)
{
    switch(_mode) {
        case Mode::Declaring: {
            add_script(
                String(boost::format("\"%1%\"") % &vertex) +
                " [label=\"=\"];\n");
            break;
        }
        case Mode::ConnectingAst: {
            add_ast_vertex(vertex, *vertex.target());
            add_ast_vertex(vertex, *vertex.expression());
            break;
        }
        case Mode::ConnectingCfg: {
            add_cfg_vertices(vertex);
            break;
        }
        case Mode::ConnectingUses: {
            break;
        }
    }

    vertex.expression()->Accept(*this);
    vertex.target()->Accept(*this);
}


void AstDotVisitor::Visit(
    OperatorVertex& vertex)
{
    switch(_mode) {
        case Mode::Declaring: {
            add_script(
                String(boost::format("\"%1%\"") % &vertex) +
                " [label=\"" + vertex.symbol() + "\"];\n"
            );
            break;
        }
        case Mode::ConnectingAst: {
            for(auto expression_vertex: vertex.expressions()) {
                add_ast_vertex(vertex, *expression_vertex);
            }
            break;
        }
        case Mode::ConnectingCfg: {
            add_cfg_vertices(vertex);
            break;
        }
        case Mode::ConnectingUses: {
            break;
        }
    }

    for(auto expression_vertex: vertex.expressions()) {
        expression_vertex->Accept(*this);
    }
}


void AstDotVisitor::Visit(
    FunctionVertex& vertex)
{
    switch(_mode) {
        case Mode::Declaring: {
            add_script(
                String(boost::format("\"%1%\"") % &vertex) +
                " [label=\"" + vertex.name() + "\"];\n");
            break;
        }
        case Mode::ConnectingAst: {
            for(auto expression_vertex: vertex.expressions()) {
                add_ast_vertex(vertex, *expression_vertex);
            }
            break;
        }
        case Mode::ConnectingCfg: {
            add_cfg_vertices(vertex);
            break;
        }
        case Mode::ConnectingUses: {
            break;
        }
    }

    for(auto expression_vertex: vertex.expressions()) {
        expression_vertex->Accept(*this);
    }
}


void AstDotVisitor::Visit(
    IfVertex& vertex)
{
    switch(_mode) {
        case Mode::Declaring: {
            add_script(
                String(boost::format("\"%1%\"") % &vertex) +
                " [label=\"If\", shape=diamond];\n"
            );
            break;
        }
        case Mode::ConnectingAst: {
            add_ast_vertex(vertex, *vertex.condition());
            for(auto statement_vertex: vertex.true_statements()) {
                add_ast_vertex(vertex, *statement_vertex);
            }
            for(auto statement_vertex: vertex.false_statements()) {
                add_ast_vertex(vertex, *statement_vertex);
            }
            break;
        }
        case Mode::ConnectingCfg: {
            add_cfg_vertices(vertex);
            break;
        }
        case Mode::ConnectingUses: {
            break;
        }
    }

    vertex.condition()->Accept(*this);
    for(auto statement_vertex: vertex.true_statements()) {
        statement_vertex->Accept(*this);
    }
    for(auto statement_vertex: vertex.false_statements()) {
        statement_vertex->Accept(*this);
    }
}


void AstDotVisitor::Visit(
    NameVertex& vertex)
{
    switch(_mode) {
        case Mode::Declaring: {
            std::vector<String> attributes;
            String label = vertex.name();

            std::vector<ExpressionVertex::ResultType> const&
                result_types(vertex.result_types());
            if(!result_types.empty()) {
                assert(result_types.size() == 1);
                String data_types = data_types_to_string(std::get<0>(
                    result_types[0]));
                String value_types = value_types_to_string(std::get<1>(
                    result_types[0]));

                label += String("\\n") +
                    "dt: " + data_types + "\\n" +
                    "vt: " + value_types;
            }

            attributes.push_back("label=\"" + label + "\"");

            add_script(
                String(boost::format("\"%1%\"") % &vertex) + " [" +
                join(attributes, ", ") + "];\n"
            );

            break;
        }
        case Mode::ConnectingAst: {
            break;
        }
        case Mode::ConnectingCfg: {
            add_cfg_vertices(vertex);
            break;
        }
        case Mode::ConnectingUses: {
            add_use_vertices(vertex);
            break;
        }
    }
}


void AstDotVisitor::Visit(
    ScriptVertex& vertex)
{
    // TODO 'ordering=out' is current not supported in combination with
    // TODO 'constraint=false'. Check again with dot > 2.28.0, when it becomes
    // TODO available.
    set_script(String(
        "digraph G {\n"
        "// ordering=out;\n"
        "rankdir=TB;\n"
    ));

    set_mode(Mode::Declaring);
    add_script(
        String(boost::format("\"%1%\"") % &vertex) +
        String(boost::format(" [label=\"%1%\"];\n")
            % vertex.source_name().encode_in_utf8())
    );

    for(auto statement_vertex: vertex.statements()) {
        statement_vertex->Accept(*this);
    }

    set_mode(Mode::ConnectingAst);
    for(auto statement_vertex: vertex.statements()) {
        add_ast_vertex(vertex, *statement_vertex);
        statement_vertex->Accept(*this);
    }

    if(_modes & Mode::ConnectingCfg) {
        set_mode(Mode::ConnectingCfg);
        add_cfg_vertices(vertex);
        for(auto statement_vertex: vertex.statements()) {
          statement_vertex->Accept(*this);
        }
    }

    if(_modes & Mode::ConnectingUses) {
        set_mode(Mode::ConnectingUses);
        for(auto statement_vertex: vertex.statements()) {
            statement_vertex->Accept(*this);
        }
    }

    add_script("}\n");
}


void AstDotVisitor::Visit(
    StringVertex& vertex)
{
    switch(_mode) {
        case Mode::Declaring: {
            add_script(
                String(boost::format("\"%1%\"") % &vertex) +
                " [label=\"\\\"" + vertex.value() +
                "\\\"\", fontname=courier, shape=box];\n"
            );
            break;
        }
        case Mode::ConnectingAst: {
            break;
        }
        case Mode::ConnectingCfg: {
            add_cfg_vertices(vertex);
            break;
        }
        case Mode::ConnectingUses: {
            break;
        }
    }
}


void AstDotVisitor::Visit(
    WhileVertex& vertex)
{
    switch(_mode) {
        case Mode::Declaring: {
            add_script(
                String(boost::format("\"%1%\"") % &vertex) +
                " [label=\"While\", shape=diamond];\n"
            );
            break;
        }
        case Mode::ConnectingAst: {
            add_ast_vertex(vertex, *vertex.condition());
            for(auto statement_vertex: vertex.true_statements()) {
                add_ast_vertex(vertex, *statement_vertex);
            }
            for(auto statement_vertex: vertex.false_statements()) {
                add_ast_vertex(vertex, *statement_vertex);
            }
            break;
        }
        case Mode::ConnectingCfg: {
            add_cfg_vertices(vertex);
            break;
        }
        case Mode::ConnectingUses: {
            break;
        }
    }

    vertex.condition()->Accept(*this);
    for(auto statement_vertex: vertex.true_statements()) {
        statement_vertex->Accept(*this);
    }
    for(auto statement_vertex: vertex.false_statements()) {
        statement_vertex->Accept(*this);
    }
}

} // namespace ranally
