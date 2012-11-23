#include "Ranally/Language/flowgraph_dot_visitor.h"
#include "Ranally/Util/string.h"
#include "Ranally/Language/vertices.h"


namespace ranally {

FlowgraphDotVisitor::FlowgraphDotVisitor()

    : DotVisitor(),
      _mode(Mode::Declaring)

{
}


void FlowgraphDotVisitor::setMode(
    Mode mode)
{
    _mode = mode;
}


void FlowgraphDotVisitor::addFlowgraphVertex(
    NameVertex const& sourceVertex,
    SyntaxVertex const& targetVertex)
{
    if(!sourceVertex.definitions().empty()) {
        for(auto definition: sourceVertex.definitions()) {
            addScript(
                String(boost::format("\"%1%\"") % definition) + " -> " +
                String(boost::format("\"%1%\"") % &targetVertex) + " ["
                "];\n"
            );
        }
    }
    else {
        addScript(
            String(boost::format("\"%1%\"") % &sourceVertex) + " -> " +
            String(boost::format("\"%1%\"") % &targetVertex) + " ["
            "];\n"
        );
    }
}


void FlowgraphDotVisitor::addFlowgraphVertex(
    SyntaxVertex const& sourceVertex,
    SyntaxVertex const& targetVertex)
{
    assert(_mode == Mode::ConnectingFlowgraph);

    if(dynamic_cast<NameVertex const*>(&sourceVertex)) {
        addFlowgraphVertex(dynamic_cast<NameVertex const&>(
            sourceVertex), targetVertex);
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
    /// const_cast<SyntaxVertex&>(sourceVertex).Accept(*this);
    /// SyntaxVertex const* newSourceVertex = _definition
    ///   ? _definition
    ///   : &sourceVertex;

    /// _script +=
    ///   String(boost::format("\"%1%\"") % newSourceVertex) + " -> " +
    ///   String(boost::format("\"%1%\"") % &targetVertex) + " ["
    ///   "];\n";

    /// _mode = Mode::ConnectingFlowgraph;
    /// _definition = 0;

    addScript(
        String(boost::format("\"%1%\"") % &sourceVertex) + " -> " +
        String(boost::format("\"%1%\"") % &targetVertex) + " ["
        "];\n"
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
            // addScript(
            //   String(boost::format("\"%1%\"") % &vertex) +
            //   " [label=\"=\"];\n");
            break;
        }
        case Mode::ConnectingFlowgraph: {
            addFlowgraphVertex(*vertex.expression(), *vertex.target());
            break;
        }
    }

    vertex.expression()->Accept(*this);
    vertex.target()->Accept(*this);
}


void FlowgraphDotVisitor::Visit(
    FunctionVertex& vertex)
{
    switch(_mode) {
        case Mode::Declaring: {
            addScript(
                String(boost::format("\"%1%\"") % &vertex) +
                " [label=\"" + vertex.name() + "\", shape=triangle];\n"
            );
            break;
        }
        case Mode::ConnectingFlowgraph: {
            for(auto expressionVertex: vertex.expressions()) {
                addFlowgraphVertex(*expressionVertex, vertex);
            }
            break;
        }
    }

    for(auto expressionVertex: vertex.expressions()) {
        expressionVertex->Accept(*this);
    }
}


void FlowgraphDotVisitor::Visit(
    IfVertex& vertex)
{
    static size_t ifClusterId = 0;
    switch(_mode) {
        case Mode::Declaring: {
            addScript(
                String(boost::format("\"%1%\"") % &vertex) +
                " [label=\"If\", shape=diamond];\n"
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
        addScript(String(boost::format(
            "subgraph cluster%1% {\n"
            "ordering=out;\n"
            "rankdir=TB;\n"
        ) % ifClusterId++));
    }
    for(auto statementVertex: vertex.trueStatements()) {
        statementVertex->Accept(*this);
    }
    if(_mode == Mode::ConnectingFlowgraph) {
        addScript("}\n");
    }

    if(!vertex.falseStatements().empty()) {
        // TODO condition -> sub graph
        if(_mode == Mode::ConnectingFlowgraph) {
            addScript(String(boost::format(
                "subgraph cluster%1% {\n"
                "ordering=out;\n"
                "rankdir=TB;\n"
            ) % ifClusterId++));
        }
        for(auto statementVertex: vertex.falseStatements()) {
            statementVertex->Accept(*this);
        }
        if(_mode == Mode::ConnectingFlowgraph) {
            addScript("}\n");
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
                addScript(
                    String(boost::format("\"%1%\"") % &vertex) +
                    " [label=\"" + vertex.name() + "\"];\n"
                );
            }
            break;
        }
        case Mode::ConnectingFlowgraph: {
            break;
        }
    }
}


template<typename T>
void FlowgraphDotVisitor::Visit(
    NumberVertex<T>& vertex)
{
    switch(_mode) {
        case Mode::Declaring: {
            addScript(
                String(boost::format("\"%1%\"") % &vertex) +
                " [label=\"" + String(boost::format("%1%") % vertex.value()) +
                "\", fontname=courier, shape=box];\n"
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
            addScript(
                String(boost::format("\"%1%\"") % &vertex) +
                " [label=\"" + vertex.symbol() + "\", shape=triangle];\n"
            );
            break;
        }
        case Mode::ConnectingFlowgraph: {
            for(auto expressionVertex: vertex.expressions()) {
                addFlowgraphVertex(*expressionVertex, vertex);
            }
            break;
        }
    }

    for(auto expressionVertex: vertex.expressions()) {
        expressionVertex->Accept(*this);
    }
}


void FlowgraphDotVisitor::Visit(
    StringVertex& vertex)
{
    switch(_mode) {
        case Mode::Declaring: {
            addScript(
                String(boost::format("\"%1%\"") % &vertex) +
                " [label=\"\\\"" + vertex.value() +
                "\\\"\", fontname=courier, shape=box];\n"
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
    ScriptVertex& vertex)
{
    setScript(String(
        "digraph G {\n"
        "ordering=out;\n"
        "rankdir=LR;\n"
        "penwidth=0.25;\n"
    ));

    setMode(Mode::Declaring);
    // addScript(
    //   String(boost::format("\"%1%\"") % &vertex) +
    //   String(boost::format(" [label=\"%1%\"];\n"))
    //     % vertex.sourceName().encodeInUTF8());

    for(auto statementVertex: vertex.statements()) {
        statementVertex->Accept(*this);
    }

    setMode(Mode::ConnectingFlowgraph);
    for(auto statementVertex: vertex.statements()) {
        // addFlowgraphVertex(vertex, *statementVertex);
        statementVertex->Accept(*this);
    }

    addScript("}\n");
}

} // namespace ranally
