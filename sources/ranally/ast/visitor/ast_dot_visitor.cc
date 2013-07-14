#include "ranally/ast/visitor/ast_dot_visitor.h"
#include "ranally/core/string.h"
#include "ranally/ast/core/vertices.h"


namespace ranally {
namespace {

String annotate_expression_label(
    String const& name,
    ExpressionVertex const& vertex)
{
    String label = name + "\\n";

    ResultTypes const& result_types(vertex.result_types());
    if(result_types.empty()) {
        label +=
            "dt: -\\n"
            "vt: -";
    }
    else {
        assert(result_types.size() == 1);
        String data_types = result_types[0].data_type().to_string();
        String value_types = result_types[0].value_type().to_string();

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
                "color=\"/spectral11/3\", "
                "constraint=false, "
                "style=dashed, "
                "penwidth=2.0"
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
              "color=\"/spectral11/8\", "
              "constraint=false, "
              "style=dashed, "
              "penwidth=2.0"
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
                String(boost::format("\"%1%\" ["
                        "label=\"%2%\", "
                        "shape=box, "
                        "style=filled, "
                        "fillcolor=\"/spectral11/9\""
                    "];\n")
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
    NumberVertex<type>& vertex)                                                \
{                                                                              \
    Visit<type>(vertex);                                                       \
}

VISIT_NUMBER_VERTICES(VISIT_NUMBER_VERTEX)

#undef VISIT_NUMBER_VERTEX


void AstDotVisitor::Visit(
    AssignmentVertex& vertex)
{
    switch(_mode) {
        case Mode::Declaring: {
            add_script(
                String(boost::format("\"%1%\" [label=\"=\"];\n") % &vertex)
            );
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
                String(boost::format("\"%1%\" ["
                        "label=\"%2%\""
                        "style=filled, "
                        "fillcolor=\"%3%\""
                    "];\n")
                    % &vertex
                    % annotate_expression_label(vertex.symbol(), vertex)
                    % (vertex.result_types().fixed()
                        ? "/spectral11/9" : "/spectral11/2")
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
                String(boost::format("\"%1%\" ["
                        "label=\"%2%\""
                        "style=filled, "
                        "fillcolor=\"%3%\""
                    "];\n")
                    % &vertex
                    % annotate_expression_label(vertex.name(), vertex)
                    % (vertex.result_types().fixed()
                        ? "/spectral11/9" : "/spectral11/2")
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
    FunctionDefinitionVertex& vertex)
{
    switch(_mode) {
        case Mode::Declaring: {
            add_script(
                String(boost::format("\"%1%\" ["
                        "label=\"%2%\" "
                    "];\n")
                    % &vertex
                    % vertex.name()));
            break;
        }
        case Mode::ConnectingAst: {
            for(auto expression_vertex: vertex.arguments()) {
                add_ast_vertex(vertex, *expression_vertex);
            }
            for(auto statement_vertex: vertex.scope()->statements()) {
                add_ast_vertex(vertex, *statement_vertex);
            }
            break;
        }
        case Mode::ConnectingCfg: {
            break;
        }
        case Mode::ConnectingUses: {
            break;
        }
    }

    for(auto expression_vertex: vertex.arguments()) {
        expression_vertex->Accept(*this);
    }
    for(auto statement_vertex: vertex.scope()->statements()) {
        statement_vertex->Accept(*this);
    }
}


void AstDotVisitor::Visit(
    ReturnVertex& vertex)
{
    switch(_mode) {
        case Mode::Declaring: {
            add_script(
                String(boost::format("\"%1%\" ["
                        "label=\"return\" "
                    "];\n")
                    % &vertex));
            break;
        }
        case Mode::ConnectingAst: {
            add_ast_vertex(vertex, *vertex.expression());
            break;
        }
        case Mode::ConnectingCfg: {
            break;
        }
        case Mode::ConnectingUses: {
            break;
        }
    }

    vertex.expression()->Accept(*this);
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
            for(auto statement_vertex: vertex.true_scope()->statements()) {
                add_ast_vertex(vertex, *statement_vertex);
            }
            for(auto statement_vertex: vertex.false_scope()->statements()) {
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
    for(auto statement_vertex: vertex.true_scope()->statements()) {
        statement_vertex->Accept(*this);
    }
    for(auto statement_vertex: vertex.false_scope()->statements()) {
        statement_vertex->Accept(*this);
    }
}


void AstDotVisitor::Visit(
    NameVertex& vertex)
{
    switch(_mode) {
        case Mode::Declaring: {
            add_script(
                String(boost::format("\"%1%\" ["
                        "label=\"%2%\" "
                        "style=filled "
                        "fillcolor=\"%3%\""
                    "];\n")
                    % &vertex
                    % annotate_expression_label(vertex.name(), vertex)
                    % (vertex.result_types().fixed()
                        ? "/spectral11/9" : "/spectral11/2")
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
                String(boost::format("\"%1%\" ["
                        "label=\"%2%\""
                        "style=filled, "
                        "fillcolor=\"%3%\""
                    "];\n")
                    % &vertex
                    % annotate_expression_label(vertex.symbol(), vertex)
                    % (vertex.result_types().fixed()
                        ? "/spectral11/9" : "/spectral11/2")
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
        "ordering=out;\n"
        "rankdir=TB;\n"
    ));

    set_mode(Mode::Declaring);
    add_script(
        String(boost::format("\"%1%\"") % &vertex) +
        String(boost::format(" [label=\"%1%\"];\n")
            % vertex.source_name().encode_in_utf8())
    );

    for(auto statement_vertex: vertex.scope()->statements()) {
        statement_vertex->Accept(*this);
    }

    set_mode(Mode::ConnectingAst);
    for(auto statement_vertex: vertex.scope()->statements()) {
        add_ast_vertex(vertex, *statement_vertex);
        statement_vertex->Accept(*this);
    }

    if(_modes & Mode::ConnectingCfg) {
        set_mode(Mode::ConnectingCfg);
        add_cfg_vertices(vertex);
        for(auto statement_vertex: vertex.scope()->statements()) {
          statement_vertex->Accept(*this);
        }
    }

    if(_modes & Mode::ConnectingUses) {
        set_mode(Mode::ConnectingUses);
        for(auto statement_vertex: vertex.scope()->statements()) {
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
            for(auto statement_vertex: vertex.true_scope()->statements()) {
                add_ast_vertex(vertex, *statement_vertex);
            }
            for(auto statement_vertex: vertex.false_scope()->statements()) {
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
    for(auto statement_vertex: vertex.true_scope()->statements()) {
        statement_vertex->Accept(*this);
    }
    for(auto statement_vertex: vertex.false_scope()->statements()) {
        statement_vertex->Accept(*this);
    }
}

} // namespace ranally
