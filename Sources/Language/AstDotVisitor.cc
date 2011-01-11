#include "AstDotVisitor.h"

#include <boost/foreach.hpp>
#include <boost/format.hpp>

#include "Vertices.h"



namespace ranally {

AstDotVisitor::AstDotVisitor()

  : DotVisitor(Ast)

{
}



AstDotVisitor::~AstDotVisitor()
{
}



// void AstDotVisitor::Visit(
//   language::AssignmentVertex& vertex)
// {
//   // assert(_mode != ConnectingOperationArgument);
//   language::ExpressionVertices const& targets = vertex.targets();
//   language::ExpressionVertices const& expressions = vertex.expressions();
// 
//   switch(mode()) {
//     case Declaring: {
//       addScript(
//         UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
//         " [label=\"=\"];\n");
//       break;
//     }
//     case ConnectingAst: {
//       assert(expressions.size() == targets.size());
//       for(size_t i = 0; i < expressions.size(); ++i) {
//         addAstVertex(vertex, *vertex.targets()[i]);
//         addAstVertex(vertex, *vertex.expressions()[i]);
//       }
//       break;
//     }
//     case ConnectingCfg: {
//       addCfgVertices(vertex);
//       break;
//     }
//     case ConnectingUses: {
//       break;
//     }
//     case ConnectingFlowgraph: {
//       break;
//     }
//     case ConnectingOperationArgument: {
//       break;
//     }
//   }
// 
//   if(mode() != ConnectingOperationArgument) {
//     BOOST_FOREACH(boost::shared_ptr<language::ExpressionVertex>
//       expressionVertex, vertex.expressions()) {
//       expressionVertex->Accept(*this);
//     }
// 
//     BOOST_FOREACH(boost::shared_ptr<language::ExpressionVertex>
//       expressionVertex, vertex.targets()) {
//       expressionVertex->Accept(*this);
//     }
//   }
// }



// void AstDotVisitor::Visit(
//   language::FunctionVertex& vertex)
// {
// }
// 
// 
// 
// void AstDotVisitor::Visit(
//   language::IfVertex& vertex)
// {
// }
// 
// 
// 
// void AstDotVisitor::Visit(
//   language::NameVertex& vertex)
// {
// }
// 
// 
// 
// void AstDotVisitor::Visit(
//   language::NumberVertex<int8_t>& vertex)
// {
// }
// 
// 
// 
// void AstDotVisitor::Visit(
//   language::NumberVertex<int16_t>& vertex)
// {
// }
// 
// 
// 
// void AstDotVisitor::Visit(
//   language::NumberVertex<int32_t>& vertex)
// {
// }
// 
// 
// 
// void AstDotVisitor::Visit(
//   language::NumberVertex<int64_t>& vertex)
// {
// }
// 
// 
// 
// void AstDotVisitor::Visit(
//   language::NumberVertex<uint8_t>& vertex)
// {
// }
// 
// 
// 
// void AstDotVisitor::Visit(
//   language::NumberVertex<uint16_t>& vertex)
// {
// }
// 
// 
// 
// void AstDotVisitor::Visit(
//   language::NumberVertex<uint32_t>& vertex)
// {
// }
// 
// 
// 
// void AstDotVisitor::Visit(
//   language::NumberVertex<uint64_t>& vertex)
// {
// }
// 
// 
// 
// void AstDotVisitor::Visit(
//   language::NumberVertex<float>& vertex)
// {
// }
// 
// 
// 
// void AstDotVisitor::Visit(
//   language::NumberVertex<double>& vertex)
// {
// }
// 
// 
// 
// void AstDotVisitor::Visit(
//   language::OperatorVertex& vertex)
// {
// }



void AstDotVisitor::Visit(
  language::ScriptVertex& vertex)
{
  setScript(UnicodeString(
    "digraph G {\n"
    "rankdir=BT;\n"
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

  setMode(ConnectingAst);
  BOOST_FOREACH(boost::shared_ptr<language::StatementVertex> statementVertex,
    vertex.statements()) {
    addAstVertex(vertex, *statementVertex);
    statementVertex->Accept(*this);
  }

  setMode(ConnectingCfg);
  addCfgVertices(vertex);
  BOOST_FOREACH(boost::shared_ptr<language::StatementVertex>
    statementVertex, vertex.statements()) {
    statementVertex->Accept(*this);
  }

  setMode(ConnectingUses);
  BOOST_FOREACH(boost::shared_ptr<language::StatementVertex>
    statementVertex, vertex.statements()) {
    statementVertex->Accept(*this);
  }

  addScript("}\n");
}



// void AstDotVisitor::Visit(
//   language::StringVertex& vertex)
// {
// }
// 
// 
// 
// void AstDotVisitor::Visit(
//   language::WhileVertex& vertex)
// {
// }

} // namespace ranally

