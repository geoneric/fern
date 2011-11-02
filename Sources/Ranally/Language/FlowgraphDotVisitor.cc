#include "FlowgraphDotVisitor.h"
#include <boost/foreach.hpp>
#include <boost/format.hpp>
#include "dev_UnicodeUtils.h"
#include "Ranally/Language/Vertices.h"



namespace ranally {

FlowgraphDotVisitor::FlowgraphDotVisitor()

  : DotVisitor(),
    _mode(Declaring)

{
}



FlowgraphDotVisitor::~FlowgraphDotVisitor()
{
}



void FlowgraphDotVisitor::setMode(
  Mode mode)
{
  _mode = mode;
}



void FlowgraphDotVisitor::addFlowgraphVertex(
  language::NameVertex const& sourceVertex,
  language::SyntaxVertex const& targetVertex)
{
  if(!sourceVertex.definitions().empty()) {
    BOOST_FOREACH(language::NameVertex* definition, sourceVertex.definitions()) {
      addScript(
        UnicodeString((boost::format("\"%1%\"")
          % definition).str().c_str()) +
        " -> " +
        (boost::format("\"%1%\"") % &targetVertex).str().c_str() + " ["
        "];\n"
      );
    }
  }
  else {
    addScript(
      UnicodeString((boost::format("\"%1%\"") % &sourceVertex).str().c_str()) +
      " -> " +
      (boost::format("\"%1%\"") % &targetVertex).str().c_str() + " ["
      "];\n"
    );
  }
}



void FlowgraphDotVisitor::addFlowgraphVertex(
  language::SyntaxVertex const& sourceVertex,
  language::SyntaxVertex const& targetVertex)
{
  assert(_mode == ConnectingFlowgraph);

  if(dynamic_cast<language::NameVertex const*>(&sourceVertex)) {
    addFlowgraphVertex(dynamic_cast<language::NameVertex const&>(sourceVertex),
      targetVertex);
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
  /// const_cast<language::SyntaxVertex&>(sourceVertex).Accept(*this);
  /// language::SyntaxVertex const* newSourceVertex = _definition
  ///   ? _definition
  ///   : &sourceVertex;

  /// _script +=
  ///   UnicodeString((boost::format("\"%1%\"") % newSourceVertex).str().c_str()) +
  ///   " -> " +
  ///   (boost::format("\"%1%\"") % &targetVertex).str().c_str() + " ["
  ///   "];\n";

  /// _mode = ConnectingFlowgraph;
  /// _definition = 0;

  addScript(
    UnicodeString((boost::format("\"%1%\"") % &sourceVertex).str().c_str()) +
    " -> " +
    (boost::format("\"%1%\"") % &targetVertex).str().c_str() + " ["
    "];\n"
  );
}



// void FlowgraphDotVisitor::addFlowgraphVertices(
//   NameVertex const& vertex)
// {
//   assert(_mode == ConnectingFlowgraph);
//   BOOST_FOREACH(NameVertex const* use, vertex.uses()) {
//     _script +=
//       UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
//       " -> " +
//       (boost::format("\"%1%\"") % use).str().c_str() + " ["
//       "];\n";
//   }
// }



void FlowgraphDotVisitor::Visit(
  language::AssignmentVertex& vertex)
{
  language::ExpressionVertices const& targets = vertex.targets();
  language::ExpressionVertices const& expressions = vertex.expressions();
  assert(expressions.size() == targets.size());

  switch(_mode) {
    case Declaring: {
      // addScript(
      //   UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
      //   " [label=\"=\"];\n");
      break;
    }
    case ConnectingFlowgraph: {
      for(size_t i = 0; i < expressions.size(); ++i) {
        addFlowgraphVertex(*vertex.expressions()[i], *vertex.targets()[i]);
      }
      break;
    }
  }

  BOOST_FOREACH(boost::shared_ptr<language::ExpressionVertex>
    expressionVertex, expressions) {
    expressionVertex->Accept(*this);
  }

  BOOST_FOREACH(boost::shared_ptr<language::ExpressionVertex>
    expressionVertex, targets) {
    expressionVertex->Accept(*this);
  }
}



void FlowgraphDotVisitor::Visit(
  language::FunctionVertex& vertex)
{
  switch(_mode) {
    case Declaring: {
      addScript(
        UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
        " [label=\"" + vertex.name() + "\", shape=triangle];\n"
      );
      break;
    }
    case ConnectingFlowgraph: {
      BOOST_FOREACH(boost::shared_ptr<language::ExpressionVertex>
        expressionVertex, vertex.expressions()) {
        addFlowgraphVertex(*expressionVertex, vertex);
      }
      break;
    }
  }

  BOOST_FOREACH(boost::shared_ptr<language::ExpressionVertex>
    expressionVertex, vertex.expressions()) {
    expressionVertex->Accept(*this);
  }
}



void FlowgraphDotVisitor::Visit(
  language::IfVertex& vertex)
{
  static size_t ifClusterId = 0;
  switch(_mode) {
    case Declaring: {
      addScript(
        UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
        " [label=\"If\", shape=diamond];\n"
      );
      break;
    }
    case ConnectingFlowgraph: {
      break;
    }
  }

  vertex.condition()->Accept(*this);

  // TODO condition -> sub graph
  if(_mode == ConnectingFlowgraph) {
    addScript(UnicodeString((boost::format(
      "subgraph cluster%1% {\n"
      "ordering=out;\n"
      "rankdir=TB;\n"
    ) % ifClusterId++).str().c_str()));
  }
  BOOST_FOREACH(boost::shared_ptr<language::StatementVertex>
    statementVertex, vertex.trueStatements()) {
    statementVertex->Accept(*this);
  }
  if(_mode == ConnectingFlowgraph) {
    addScript("}\n");
  }

  if(!vertex.falseStatements().empty()) {
    // TODO condition -> sub graph
    if(_mode == ConnectingFlowgraph) {
      addScript(UnicodeString((boost::format(
        "subgraph cluster%1% {\n"
        "ordering=out;\n"
        "rankdir=TB;\n"
      ) % ifClusterId++).str().c_str()));
    }
    BOOST_FOREACH(boost::shared_ptr<language::StatementVertex>
      statementVertex, vertex.falseStatements()) {
      statementVertex->Accept(*this);
    }
    if(_mode == ConnectingFlowgraph) {
      addScript("}\n");
    }
  }
}



void FlowgraphDotVisitor::Visit(
  language::NameVertex& vertex)
{
  switch(_mode) {
    case Declaring: {
      // Filter out those identifiers that are defined somewhere else. Declare
      // only those identifiers that have no definition, or that are part of
      // the defining expression.
      if(vertex.definitions().empty() || std::find(vertex.definitions().begin(),
        vertex.definitions().end(), &vertex) != vertex.definitions().end()) {
        addScript(
          UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
          " [label=\"" + vertex.name() + "\"];\n"
        );
      }
      break;
    }
    case ConnectingFlowgraph: {
      break;
    }
  }
}



template<typename T>
void FlowgraphDotVisitor::Visit(
  language::NumberVertex<T>& vertex)
{
  switch(_mode) {
    case Declaring: {
      addScript(
        UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
        " [label=\"" + (boost::format("%1%") % vertex.value()).str().c_str() +
        "\", fontname=courier, shape=box];\n"
      );
      break;
    }
    case ConnectingFlowgraph: {
      break;
    }
  }
}



void FlowgraphDotVisitor::Visit(
  language::NumberVertex<int8_t>& vertex)
{
  return Visit<int8_t>(vertex);
}



void FlowgraphDotVisitor::Visit(
  language::NumberVertex<int16_t>& vertex)
{
  return Visit<int16_t>(vertex);
}



void FlowgraphDotVisitor::Visit(
  language::NumberVertex<int32_t>& vertex)
{
  return Visit<int32_t>(vertex);
}



void FlowgraphDotVisitor::Visit(
  language::NumberVertex<int64_t>& vertex)
{
  return Visit<int64_t>(vertex);
}



void FlowgraphDotVisitor::Visit(
  language::NumberVertex<uint8_t>& vertex)
{
  return Visit<uint8_t>(vertex);
}



void FlowgraphDotVisitor::Visit(
  language::NumberVertex<uint16_t>& vertex)
{
  return Visit<uint16_t>(vertex);
}



void FlowgraphDotVisitor::Visit(
  language::NumberVertex<uint32_t>& vertex)
{
  return Visit<uint32_t>(vertex);
}



void FlowgraphDotVisitor::Visit(
  language::NumberVertex<uint64_t>& vertex)
{
  return Visit<uint64_t>(vertex);
}



void FlowgraphDotVisitor::Visit(
  language::NumberVertex<float>& vertex)
{
  return Visit<float>(vertex);
}



void FlowgraphDotVisitor::Visit(
  language::NumberVertex<double>& vertex)
{
  return Visit<double>(vertex);
}


void FlowgraphDotVisitor::Visit(
  language::OperatorVertex& vertex)
{
  switch(_mode) {
    case Declaring: {
      // TODO Implement symbol member.
      addScript(
        UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
        " [label=\"" + vertex.symbol() + "\", shape=triangle];\n"
      );
      break;
    }
    case ConnectingFlowgraph: {
      BOOST_FOREACH(boost::shared_ptr<language::ExpressionVertex>
        expressionVertex, vertex.expressions()) {
        addFlowgraphVertex(*expressionVertex, vertex);
      }
      break;
    }
  }

  BOOST_FOREACH(boost::shared_ptr<language::ExpressionVertex>
    expressionVertex, vertex.expressions()) {
    expressionVertex->Accept(*this);
  }
}



void FlowgraphDotVisitor::Visit(
  language::StringVertex& vertex)
{
  switch(_mode) {
    case Declaring: {
      addScript(
        UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
        " [label=\"\\\"" + vertex.value() +
        "\\\"\", fontname=courier, shape=box];\n"
      );
      break;
    }
    case ConnectingFlowgraph: {
      break;
    }
  }
}



void FlowgraphDotVisitor::Visit(
  language::WhileVertex& /* vertex */)
{
}



void FlowgraphDotVisitor::Visit(
  language::ScriptVertex& vertex)
{
  setScript(UnicodeString(
    "digraph G {\n"
    "ordering=out;\n"
    "rankdir=LR;\n"
    "penwidth=0.25;\n"
  ));

  setMode(Declaring);
  // addScript(
  //   UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
  //   (boost::format(" [label=\"%1%\"];\n")
  //     % dev::encodeInUTF8(vertex.sourceName())).str().c_str());

  BOOST_FOREACH(boost::shared_ptr<language::StatementVertex> statementVertex,
    vertex.statements()) {
    statementVertex->Accept(*this);
  }

  setMode(ConnectingFlowgraph);
  BOOST_FOREACH(boost::shared_ptr<language::StatementVertex> statementVertex,
    vertex.statements()) {
    // addFlowgraphVertex(vertex, *statementVertex);
    statementVertex->Accept(*this);
  }

  addScript("}\n");
}

} // namespace ranally

