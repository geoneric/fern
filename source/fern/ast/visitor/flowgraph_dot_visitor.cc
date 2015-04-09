// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/ast/visitor/flowgraph_dot_visitor.h"
#include "fern/core/string.h"
#include "fern/ast/core/vertices.h"


namespace fern {

FlowgraphDotVisitor::FlowgraphDotVisitor()

    : DotVisitor(),
      _mode(Mode::Declaring)

{
}


void FlowgraphDotVisitor::set_mode(
    Mode mode)
{
    _mode = mode;
}


void FlowgraphDotVisitor::add_flowgraph_vertex(
    NameVertex const& source_vertex,
    AstVertex const& target_vertex)
{
    if(!source_vertex.definitions().empty()) {
        for(auto definition: source_vertex.definitions()) {
            add_script(
                String(boost::format("\"%1%\"") % definition) +
                String(" -> ") +
                String(boost::format("\"%1%\"") % &target_vertex) +
                String(" [];\n")
            );
        }
    }
    else {
        add_script(
            String(boost::format("\"%1%\"") % &source_vertex) +
            String(" -> ") +
            String(boost::format("\"%1%\"") % &target_vertex) +
            String(" [];\n")
        );
    }
}


void FlowgraphDotVisitor::add_flowgraph_vertex(
    AstVertex const& source_vertex,
    AstVertex const& target_vertex)
{
    assert(_mode == Mode::ConnectingFlowgraph);

    if(dynamic_cast<NameVertex const*>(&source_vertex)) {
        add_flowgraph_vertex(dynamic_cast<NameVertex const&>(
            source_vertex), target_vertex);
        return;
    }

    /// // In case the operation argument is a NameVertex, we want to connect the
    /// // defining location to the target. Let's find the definition. This only
    /// // succeeds if the source vertex is a NameVertex and if it has a definition.
    /// // TODO Can't we depend on preprocessing (clean up) of the tree, instead of
    /// //      diving in? We may want to show the current state of the tree, whatever
    /// //      it is, instead of optimizing the plot by finding stuff ourselves.
    /// assert(!_definition);
    /// _mode = ConnectingOperationArgument;
    /// const_cast<AstVertex&>(source_vertex).Accept(*this);
    /// AstVertex const* new_source_vertex = _definition
    ///   ? _definition
    ///   : &source_vertex;

    /// _script +=
    ///   String(boost::format("\"%1%\"") % new_source_vertex) + " -> " +
    ///   String(boost::format("\"%1%\"") % &target_vertex) + " ["
    ///   "];\n";

    /// _mode = Mode::ConnectingFlowgraph;
    /// _definition = 0;

    add_script(
        String(boost::format("\"%1%\"") % &source_vertex) +
        String(" -> ") +
        String(boost::format("\"%1%\"") % &target_vertex) +
        String(" [];\n")
    );
}


// void FlowgraphDotVisitor::addFlowgraphVertices(
//   NameVertex const& vertex)
// {
//   assert(_mode == Mode::ConnectingFlowgraph);
//   BOOST_FOREACH(NameVertex const* use, vertex.uses()) {
//     _script +=
//       String(boost::format("\"%1%\"") % &vertex) + " -> " +
//       String(boost::format("\"%1%\"") % use) + " ["
//       "];\n";
//   }
// }


void FlowgraphDotVisitor::Visit(
    AssignmentVertex& vertex)
{
    switch(_mode) {
        case Mode::Declaring: {
            // add_script(
            //   String(boost::format("\"%1%\"") % &vertex) +
            //   " [label=\"=\"];\n");
            break;
        }
        case Mode::ConnectingFlowgraph: {
            add_flowgraph_vertex(*vertex.expression(), *vertex.target());
            break;
        }
    }

    vertex.expression()->Accept(*this);
    vertex.target()->Accept(*this);
}


void FlowgraphDotVisitor::Visit(
    FunctionCallVertex& vertex)
{
    switch(_mode) {
        case Mode::Declaring: {
            add_script(
                String(boost::format("\"%1%\"") % &vertex) +
                String(" [label=\"") + vertex.name() +
                String("\", shape=triangle];\n")
            );
            break;
        }
        case Mode::ConnectingFlowgraph: {
            for(auto expression_vertex: vertex.expressions()) {
                add_flowgraph_vertex(*expression_vertex, vertex);
            }
            break;
        }
    }

    for(auto expression_vertex: vertex.expressions()) {
        expression_vertex->Accept(*this);
    }
}


void FlowgraphDotVisitor::Visit(
    IfVertex& vertex)
{
    static size_t if_cluster_id = 0;
    switch(_mode) {
        case Mode::Declaring: {
            add_script(
                String(boost::format("\"%1%\"") % &vertex) +
                String(" [label=\"If\", shape=diamond];\n")
            );
            break;
        }
        case Mode::ConnectingFlowgraph: {
            break;
        }
    }

    vertex.condition()->Accept(*this);

    // TODO condition -> sub graph
    if(_mode == Mode::ConnectingFlowgraph) {
        add_script(String(boost::format(
            "subgraph cluster%1% {\n"
            "ordering=out;\n"
            "rankdir=TB;\n"
        ) % if_cluster_id++));
    }
    for(auto statement_vertex: vertex.true_scope()->statements()) {
        statement_vertex->Accept(*this);
    }
    if(_mode == Mode::ConnectingFlowgraph) {
        add_script("}\n");
    }

    if(!vertex.false_scope()->statements().empty()) {
        // TODO condition -> sub graph
        if(_mode == Mode::ConnectingFlowgraph) {
            add_script(String(boost::format(
                "subgraph cluster%1% {\n"
                "ordering=out;\n"
                "rankdir=TB;\n"
            ) % if_cluster_id++));
        }
        for(auto statement_vertex: vertex.false_scope()->statements()) {
            statement_vertex->Accept(*this);
        }
        if(_mode == Mode::ConnectingFlowgraph) {
            add_script("}\n");
        }
    }
}


void FlowgraphDotVisitor::Visit(
    NameVertex& vertex)
{
    switch(_mode) {
        case Mode::Declaring: {
            // Filter out those identifiers that are defined somewhere
            // else. Declare only those identifiers that have no definition,
            // or that are part of the defining expression.
            if(vertex.definitions().empty() ||
                    std::find(vertex.definitions().begin(),
                        vertex.definitions().end(), &vertex) !=
                    vertex.definitions().end()) {
                add_script(
                    String(boost::format("\"%1%\"") % &vertex) +
                    String(" [label=\"") + vertex.name() + String("\"];\n")
                );
            }
            break;
        }
        case Mode::ConnectingFlowgraph: {
            break;
        }
    }
}


void FlowgraphDotVisitor::Visit(
    SubscriptVertex& vertex)
{
    switch(_mode) {
        case Mode::Declaring: {
            add_script(
                String(boost::format("\"%1%\"") % &vertex) +
                String(" [label=\"") + vertex.symbol() +
                String("\", shape=triangle];\n")
            );
            break;
        }
        case Mode::ConnectingFlowgraph: {
            add_flowgraph_vertex(*vertex.expression(), vertex);
            add_flowgraph_vertex(*vertex.selection(), vertex);
            break;
        }
    }

    vertex.expression()->Accept(*this);
    vertex.selection()->Accept(*this);
}


template<typename T>
void FlowgraphDotVisitor::Visit(
    NumberVertex<T>& vertex)
{
    switch(_mode) {
        case Mode::Declaring: {
            add_script(
                String(boost::format("\"%1%\"") % &vertex) +
                String(" [label=\"") +
                String(boost::format("%1%") % vertex.value()) +
                String("\", fontname=courier, shape=box];\n")
            );
            break;
        }
        case Mode::ConnectingFlowgraph: {
            break;
        }
    }
}


void FlowgraphDotVisitor::Visit(
    NumberVertex<int8_t>& vertex)
{
    return Visit<int8_t>(vertex);
}


void FlowgraphDotVisitor::Visit(
    NumberVertex<int16_t>& vertex)
{
    return Visit<int16_t>(vertex);
}


void FlowgraphDotVisitor::Visit(
    NumberVertex<int32_t>& vertex)
{
    return Visit<int32_t>(vertex);
}


void FlowgraphDotVisitor::Visit(
    NumberVertex<int64_t>& vertex)
{
    return Visit<int64_t>(vertex);
}


void FlowgraphDotVisitor::Visit(
    NumberVertex<uint8_t>& vertex)
{
    return Visit<uint8_t>(vertex);
}


void FlowgraphDotVisitor::Visit(
    NumberVertex<uint16_t>& vertex)
{
    return Visit<uint16_t>(vertex);
}


void FlowgraphDotVisitor::Visit(
    NumberVertex<uint32_t>& vertex)
{
    return Visit<uint32_t>(vertex);
}


void FlowgraphDotVisitor::Visit(
    NumberVertex<uint64_t>& vertex)
{
    return Visit<uint64_t>(vertex);
}


void FlowgraphDotVisitor::Visit(
    NumberVertex<float>& vertex)
{
    return Visit<float>(vertex);
}


void FlowgraphDotVisitor::Visit(
    NumberVertex<double>& vertex)
{
    return Visit<double>(vertex);
}


void FlowgraphDotVisitor::Visit(
    OperatorVertex& vertex)
{
    switch(_mode) {
        case Mode::Declaring: {
            // TODO Implement symbol member.
            add_script(
                String(boost::format("\"%1%\"") % &vertex) +
                String(" [label=\"") + vertex.symbol() +
                String("\", shape=triangle];\n")
            );
            break;
        }
        case Mode::ConnectingFlowgraph: {
            for(auto expression_vertex: vertex.expressions()) {
                add_flowgraph_vertex(*expression_vertex, vertex);
            }
            break;
        }
    }

    for(auto expression_vertex: vertex.expressions()) {
        expression_vertex->Accept(*this);
    }
}


void FlowgraphDotVisitor::Visit(
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
        case Mode::ConnectingFlowgraph: {
            break;
        }
    }
}


void FlowgraphDotVisitor::Visit(
    WhileVertex& /* vertex */)
{
}


void FlowgraphDotVisitor::Visit(
    ModuleVertex& vertex)
{
    set_script(String(
        "digraph G {\n"
        "ordering=out;\n"
        "rankdir=LR;\n"
        "penwidth=0.25;\n"
    ));

    set_mode(Mode::Declaring);
    // add_script(
    //   String(boost::format("\"%1%\"") % &vertex) +
    //   String(boost::format(" [label=\"%1%\"];\n"))
    //     % vertex.sourceName().encode_in_utf8());

    for(auto statement_vertex: vertex.scope()->statements()) {
        statement_vertex->Accept(*this);
    }

    set_mode(Mode::ConnectingFlowgraph);
    for(auto statement_vertex: vertex.scope()->statements()) {
        // add_flowgraph_vertex(vertex, *statement_vertex);
        statement_vertex->Accept(*this);
    }

    add_script("}\n");
}

} // namespace fern
