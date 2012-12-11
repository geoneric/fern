#include "ranally/language/ast_dot_visitor.h"
#include "ranally/core/string.h"
#include "ranally/language/vertices.h"


namespace ranally {
namespace {

String annotate_expression_label(
    String const& name,
    ExpressionVertex const& vertex)
{
    String label = name + "\\n";

    std::vector<ExpressionVertex::ResultType> const&
        result_types(vertex.result_types());
    if(result_types.empty()) {
        label +=
            "dt: unknown\\n"
            "vt: unknown";
    }
    else {
        assert(result_types.size() == 1);
        String data_types = std::get<0>(result_types[0]).to_string();
        String value_types = std::get<1>(result_types[0]).to_string();

        label +=
            "dt: " + data_types + "\\n"
            "vt: " + value_types;
    }

    return label;
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
                String(boost::format("\"%1%\" [label=\"%2%\", shape=box];\n")
                    % &vertex
                    % annotate_expression_label(
                        (boost::format("%1%")) % vertex.value(), vertex)
            ));

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
                String(boost::format("\"%1%\" [label=\"%2%\"];\n")
                    % &vertex
                    % annotate_expression_label(vertex.symbol(), vertex)
            ));
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
                String(boost::format("\"%1%\" [label=\"%2%\"];\n")
                    % &vertex
                    % annotate_expression_label(vertex.name(), vertex)
            ));
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
            add_script(
                String(boost::format("\"%1%\" [label=\"%2%\"];\n")
                    % &vertex
                    % annotate_expression_label(vertex.name(), vertex)
            ));

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
    SubscriptVertex& vertex)
{
    switch(_mode) {
        case Mode::Declaring: {
            add_script(
                String(boost::format("\"%1%\" [label=\"%2%\"];\n")
                    % &vertex
                    % annotate_expression_label(vertex.symbol(), vertex)
            ));
            break;
        }
        case Mode::ConnectingAst: {
            add_ast_vertex(vertex, *vertex.expression());
            add_ast_vertex(vertex, *vertex.selection());
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
    vertex.selection()->Accept(*this);
}


void AstDotVisitor::Visit(
    ScriptVertex& vertex)
{
    // TODO 'ordering=out' is current not supported in combination with
    // TODO 'constraint=false'. Check again with dot > 2.28.0, when it becomes
    // TODO available.
    set_script(String(
        "digraph G {\n"
        // "ordering=out;\n"
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
