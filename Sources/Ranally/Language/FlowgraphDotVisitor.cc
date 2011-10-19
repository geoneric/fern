#include "FlowgraphDotVisitor.h"
#include <boost/foreach.hpp>
#include <boost/format.hpp>
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
  language::SyntaxVertex const& /* sourceVertex */,
  language::SyntaxVertex const& /* targetVertex */)
{
  /// assert(_mode == ConnectingFlowgraph);

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
}



void FlowgraphDotVisitor::Visit(
  language::FunctionVertex& vertex)
{
}



void FlowgraphDotVisitor::Visit(
  language::IfVertex& vertex)
{
}



void FlowgraphDotVisitor::Visit(
  language::NameVertex& vertex)
{
}



void FlowgraphDotVisitor::Visit(
  language::NumberVertex<int8_t>& vertex)
{
}



void FlowgraphDotVisitor::Visit(
  language::NumberVertex<int16_t>& vertex)
{
}



void FlowgraphDotVisitor::Visit(
  language::NumberVertex<int32_t>& vertex)
{
}



void FlowgraphDotVisitor::Visit(
  language::NumberVertex<int64_t>& vertex)
{
}



void FlowgraphDotVisitor::Visit(
  language::NumberVertex<uint8_t>& vertex)
{
}



void FlowgraphDotVisitor::Visit(
  language::NumberVertex<uint16_t>& vertex)
{
}



void FlowgraphDotVisitor::Visit(
  language::NumberVertex<uint32_t>& vertex)
{
}



void FlowgraphDotVisitor::Visit(
  language::NumberVertex<uint64_t>& vertex)
{
}



void FlowgraphDotVisitor::Visit(
  language::NumberVertex<float>& vertex)
{
}



void FlowgraphDotVisitor::Visit(
  language::NumberVertex<double>& vertex)
{
}


void FlowgraphDotVisitor::Visit(
  language::OperatorVertex& vertex)
{
}



void FlowgraphDotVisitor::Visit(
  language::StringVertex& vertex)
{
}



void FlowgraphDotVisitor::Visit(
  language::WhileVertex& vertex)
{
}



void FlowgraphDotVisitor::Visit(
  language::ScriptVertex& vertex)
{
  setScript(UnicodeString(
    "digraph G {\n"
    "ordering=out;\n"
    "rankdir=TB;\n"
  ));

  setMode(Declaring);
  // TODO Use script name.
  addScript(
    UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
    " [label=\"Script\"];\n");

  BOOST_FOREACH(boost::shared_ptr<language::StatementVertex> statementVertex,
    vertex.statements()) {
    statementVertex->Accept(*this);
  }

  // setMode(ConnectingFlowgraph);
  // BOOST_FOREACH(boost::shared_ptr<language::StatementVertex> statementVertex,
  //   vertex.statements()) {
  //   addAstVertex(vertex, *statementVertex);
  //   statementVertex->Accept(*this);
  // }

  addScript("}\n");
}

} // namespace ranally

