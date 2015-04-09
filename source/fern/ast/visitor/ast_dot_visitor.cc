// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/ast/visitor/ast_dot_visitor.h"
#include "fern/core/string.h"
#include "fern/ast/core/vertices.h"


namespace fern {
namespace {

String annotate_expression_label(
    String const& name,
    ExpressionVertex const& vertex)
{
    String label = name + String("\\n");

    ExpressionTypes const& expression_types(vertex.expression_types());
    if(expression_types.is_empty()) {
        label +=
            "dt: -\\n"
            "vt: -";
    }
    else {
        assert(expression_types.size() == 1);
        String data_types = expression_types[0].data_type().to_string();
        String value_types = expression_types[0].value_type().to_string();

        label +=
            String("dt: ") + data_types + String("\\n") +
            String("vt: ") + value_types;
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
    AstVertex const& source_vertex,
    AstVertex const& target_vertex)
{
    assert(_mode == Mode::ConnectingAst);
    add_script(
        String(boost::format("\"%1%\"") % &source_vertex) + String(" -> ") +
        String(boost::format("\"%1%\"") % &target_vertex) + String(" [") +
        String("];\n")
    );
}


// void AstDotVisitor::add_ast_vertex(
//     AstVertex const& source_vertex,
//     ScopeVertex const& target_vertex)
// {
//     assert(_mode == Mode::ConnectingAst);
//     add_script(
//         String(boost::format("\"%1%\"") % &source_vertex) + " -> " +
//         String(boost::format("\"%1%\"") % &target_vertex) + " ["
//         // String(boost::format("\"cluster_%1%\"") % &target_vertex) + " ["
//         "];\n"
//     );
// }
// 
// 
// void AstDotVisitor::add_ast_vertex(
//     ScopeVertex const& source_vertex,
//     AstVertex const& target_vertex)
// {
//     assert(_mode == Mode::ConnectingAst);
//     add_script(
//         // String(boost::format("\"cluster_%1%\"") % &source_vertex) + " -> " +
//         String(boost::format("\"%1%\"") % &source_vertex) + " -> " +
//         String(boost::format("\"%1%\"") % &target_vertex) + " ["
//         "];\n"
//     );
// }


void AstDotVisitor::add_cfg_vertices(
    AstVertex const& source_vertex)
{
    assert(_mode == Mode::ConnectingCfg);
    for(auto successor: source_vertex.successors()) {
        add_script(
            String(boost::format("\"%1%\"") % &source_vertex) + String(" -> ") +
            String(boost::format("\"%1%\"") % successor) + String(" [") +
                String(
                    "color=\"/spectral11/3\", "
                    "constraint=false, "
                    "style=dashed, "
                    "penwidth=2.0"
                "];\n")
        );
    }
}


void AstDotVisitor::add_use_vertices(
    NameVertex const& vertex)
{
    assert(_mode == Mode::ConnectingUses);
    for(auto use: vertex.uses()) {
        add_script(
            String(boost::format("\"%1%\"") % &vertex) + String(" -> ") +
            String(boost::format("\"%1%\"") % use) + String(" [") +
              String(
                  "color=\"/spectral11/8\", "
                  "constraint=false, "
                  "style=dashed, "
                  "penwidth=2.0"
              "];\n")
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
                    % (vertex.expression_types().fixed()
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
    FunctionCallVertex& vertex)
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
                    % (vertex.expression_types().fixed()
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
    _scope_names.push(vertex.name().encode_in_utf8());

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
            // Visit(*vertex.scope());
            // add_script(String(boost::format(
            //     "subgraph cluster_%1% {\n"
            //         "label=\"%2%\";\n")
            //     % &vertex
            //     % vertex.name()));

            // for(auto expression_vertex: vertex.arguments()) {
            //     add_ast_vertex(vertex, *expression_vertex);
            // }
            // for(auto statement_vertex: vertex.scope()->statements()) {
            //     add_ast_vertex(vertex, *statement_vertex);
            // }

            // for(auto expression_vertex: vertex.arguments()) {
            //     expression_vertex->Accept(*this);
            // }
            // for(auto statement_vertex: vertex.scope()->statements()) {
            //     statement_vertex->Accept(*this);
            // }

            // add_script("}\n");
            break;
        }
        case Mode::ConnectingCfg: {
            break;
        }
        case Mode::ConnectingUses: {
            break;
        }
    }

    // for(auto expression_vertex: vertex.arguments()) {
    //     expression_vertex->Accept(*this);
    // }
    Visit(*vertex.scope());
    // vertex.scope()->Accept(*this);

    // // // Recursing AST in case we are connecting is already handled above.
    // // if(_mode != Mode::ConnectingAst) {
    //     for(auto expression_vertex: vertex.arguments()) {
    //         expression_vertex->Accept(*this);
    //     }
    //     // for(auto statement_vertex: vertex.scope()->statements()) {
    //     //     statement_vertex->Accept(*this);
    //     // }
    //     vertex.scope()->Accept(*this);
    // // }
    _scope_names.pop();
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
            if(vertex.expression()) {
                add_ast_vertex(vertex, *vertex.expression());
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

    if(vertex.expression()) {
        vertex.expression()->Accept(*this);
    }
}


void AstDotVisitor::Visit(
    IfVertex& vertex)
{
    switch(_mode) {
        case Mode::Declaring: {
            add_script(
                String(boost::format("\"%1%\"") % &vertex) +
                String(" [label=\"If\", shape=diamond];\n")
            );
            break;
        }
        case Mode::ConnectingAst: {
            add_ast_vertex(vertex, *vertex.condition());
            // for(auto statement_vertex: vertex.true_scope()->statements()) {
            //     add_ast_vertex(vertex, *statement_vertex);
            // }
            // for(auto statement_vertex: vertex.false_scope()->statements()) {
            //     add_ast_vertex(vertex, *statement_vertex);
            // }
            break;
        }
        case Mode::ConnectingCfg: {
            // add_cfg_vertices(vertex);
            break;
        }
        case Mode::ConnectingUses: {
            break;
        }
    }

    vertex.condition()->Accept(*this);
    vertex.true_scope()->Accept(*this);
    vertex.false_scope()->Accept(*this);
    // for(auto statement_vertex: vertex.true_scope()->statements()) {
    //     statement_vertex->Accept(*this);
    // }
    // for(auto statement_vertex: vertex.false_scope()->statements()) {
    //     statement_vertex->Accept(*this);
    // }
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
                    % (vertex.expression_types().fixed()
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
                    % (vertex.expression_types().fixed()
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
    ModuleVertex& vertex)
{
    _scope_names.push(vertex.source_name().encode_in_utf8());
    // TODO 'ordering=out' is current not supported in combination with
    // TODO 'constraint=false'. Check again with dot > 2.28.0, when it becomes
    // TODO available.
    set_script(String(
        "digraph G {\n"
        "ordering=out;\n"
        "rankdir=TB;\n"
    ));

    set_mode(Mode::Declaring);
    Visit(*vertex.scope());

    set_mode(Mode::ConnectingAst);
    Visit(*vertex.scope());

    if(_modes & Mode::ConnectingCfg) {
        set_mode(Mode::ConnectingCfg);
        // add_cfg_vertices(vertex);
        Visit(*vertex.scope());
        // vertex.scope()->Accept(*this);
        // for(auto statement_vertex: vertex.scope()->statements()) {
        //   statement_vertex->Accept(*this);
        // }
    }

    if(_modes & Mode::ConnectingUses) {
        set_mode(Mode::ConnectingUses);
        // vertex.scope()->Accept(*this);
        Visit(*vertex.scope());
        // for(auto statement_vertex: vertex.scope()->statements()) {
        //     statement_vertex->Accept(*this);
        // }
    }

    add_script("}\n");
    _scope_names.pop();
}


void AstDotVisitor::Visit(
    StringVertex& vertex)
{
    switch(_mode) {
        case Mode::Declaring: {
            add_script(
                String(boost::format("\"%1%\"") % &vertex) +
                String(" [label=\"\\\"") + vertex.value() +
                String("\\\"\", fontname=courier, shape=box];\n")
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
                String(" [label=\"While\", shape=diamond];\n")
            );
            break;
        }
        case Mode::ConnectingAst: {
            add_ast_vertex(vertex, *vertex.condition());
            // for(auto statement_vertex: vertex.true_scope()->statements()) {
            //     add_ast_vertex(vertex, *statement_vertex);
            // }
            // for(auto statement_vertex: vertex.false_scope()->statements()) {
            //     add_ast_vertex(vertex, *statement_vertex);
            // }
            break;
        }
        case Mode::ConnectingCfg: {
            // add_cfg_vertices(vertex);
            break;
        }
        case Mode::ConnectingUses: {
            break;
        }
    }

    vertex.condition()->Accept(*this);
    vertex.true_scope()->Accept(*this);
    vertex.false_scope()->Accept(*this);
    // for(auto statement_vertex: vertex.true_scope()->statements()) {
    //     statement_vertex->Accept(*this);
    // }
    // for(auto statement_vertex: vertex.false_scope()->statements()) {
    //     statement_vertex->Accept(*this);
    // }
}


void AstDotVisitor::Visit(
    ScopeVertex& vertex)
{
    switch(_mode) {
        case Mode::Declaring: {
            add_script(
                String(boost::format("\"%1%\" ["
                        "label=\"scope\" "
                    "];\n")
                    % &vertex));
            break;
        }
        case Mode::ConnectingAst: {
            add_script(String(boost::format(
                "subgraph cluster_%1% {\n"
                    "label=\"%2%\";\n")
                % &vertex
                % _scope_names.top()));

            for(auto statement_vertex: vertex.statements()) {
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

    for(auto statement_vertex: vertex.statements()) {
        statement_vertex->Accept(*this);
    }
    Visit(*vertex.sentinel());
}


void AstDotVisitor::Visit(
    SentinelVertex& vertex)
{
    switch(_mode) {
        case Mode::Declaring: {
            add_script(
                String(boost::format("\"%1%\" ["
                        "label=\"sentinel\" "
                    "];\n")
                    % &vertex));
            break;
        }
        case Mode::ConnectingAst: {
            add_script("}\n");
            break;
        }
        case Mode::ConnectingCfg: {
            break;
        }
        case Mode::ConnectingUses: {
            break;
        }
    }
}

} // namespace fern
