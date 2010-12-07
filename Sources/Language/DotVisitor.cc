#include "DotVisitor.h"

#include <boost/foreach.hpp>
#include <boost/format.hpp>

#include "Vertices.h"



namespace ranally {

DotVisitor::DotVisitor()

  : _type(Flowgraph),
    _mode(Declaring)

{
}



DotVisitor::~DotVisitor()
{
}



UnicodeString const& DotVisitor::script() const
{
  return _script;
}



void DotVisitor::addAstVertex(
  SyntaxVertex const& sourceVertex,
  SyntaxVertex const& targetVertex)
{
  assert(_mode == ConnectingAst);
  _script +=
    UnicodeString((boost::format("\"%1%\"") % &sourceVertex).str().c_str()) +
    " -> " +
    (boost::format("\"%1%\"") % &targetVertex).str().c_str() + " ["
    "];\n";
}



void DotVisitor::addCfgVertices(
  SyntaxVertex const& sourceVertex)
{
  assert(_mode == ConnectingCfg);
  BOOST_FOREACH(SyntaxVertex const* successor, sourceVertex.successors()) {
    _script +=
      UnicodeString((boost::format("\"%1%\"") % &sourceVertex).str().c_str()) +
      " -> " +
      (boost::format("\"%1%\"") % successor).str().c_str() + " ["
        "color=red, "
        "constraint=false, "
        "style=dashed"
      "];\n";
  }
}



void DotVisitor::addUseVertices(
  NameVertex const& vertex)
{
  assert(_mode == ConnectingUses);
  BOOST_FOREACH(NameVertex const* use, vertex.uses()) {
    _script +=
      UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
      " -> " +
      (boost::format("\"%1%\"") % use).str().c_str() + " ["
        "color=blue, "
        "constraint=false, "
        "style=dotted"
      "];\n";
  }
}



void DotVisitor::Visit(
  AssignmentVertex& vertex)
{
  ExpressionVertices const& targets = vertex.targets();
  ExpressionVertices const& expressions = vertex.expressions();

  switch(_mode) {
    case Declaring: {
      _script +=
        UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
        " [label=\"=\"];\n";
      break;
    }
    case ConnectingAst: {
      assert(expressions.size() == targets.size());
      for(size_t i = 0; i < expressions.size(); ++i) {
        addAstVertex(vertex, *vertex.targets()[i]);
        addAstVertex(vertex, *vertex.expressions()[i]);
      }
      break;
    }
    case ConnectingCfg: {
      addCfgVertices(vertex);
      break;
    }
    case ConnectingUses: {
      break;
    }
    case ConnectingFlowgraph: {
      // hier verder
      // a = b + c
      // d = f(a)

      // b -> +
      // c -> +
      // + -> f
      //
      // Prereq: For each name it must be known what the address is of the
      //         defining expression.
      // 1. defining expression of operands -> operator
      // 2. defining expression of arguments -> functions
      //
      //
      // assert(expressions.size() == targets.size());
      // for(size_t i = 0; i < expressions.size(); ++i) {
      //   NameVertex* nameVertex = dynamic_cast<NameVertex*>(targets[i]);
      //   assert(nameVertex);
      // }

      break;
    }
  }

  BOOST_FOREACH(boost::shared_ptr<ranally::ExpressionVertex> expressionVertex,
    vertex.expressions()) {
    expressionVertex->Accept(*this);
  }

  BOOST_FOREACH(boost::shared_ptr<ranally::ExpressionVertex> expressionVertex,
    vertex.targets()) {
    expressionVertex->Accept(*this);
  }
}



void DotVisitor::Visit(
  FunctionVertex& vertex)
{
  switch(_mode) {
    case Declaring: {
      _script +=
        UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
        " [label=\"" + vertex.name() + "\"];\n";
      break;
    }
    case ConnectingAst: {
      BOOST_FOREACH(boost::shared_ptr<ranally::ExpressionVertex>
        expressionVertex, vertex.expressions()) {
        addAstVertex(vertex, *expressionVertex);
      }
      break;
    }
    case ConnectingCfg: {
      addCfgVertices(vertex);
      break;
    }
    case ConnectingUses: {
      break;
    }
  }

  BOOST_FOREACH(boost::shared_ptr<ranally::ExpressionVertex>
    expressionVertex, vertex.expressions()) {
    expressionVertex->Accept(*this);
  }
}



void DotVisitor::Visit(
  OperatorVertex& vertex)
{
  switch(_mode) {
    case Declaring: {
      // TODO Implement symbol member.
      _script +=
        UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
        " [label=\"" + vertex.symbol() + "\"];\n";
      break;
    }
    case ConnectingAst: {
      BOOST_FOREACH(boost::shared_ptr<ranally::ExpressionVertex>
        expressionVertex, vertex.expressions()) {
        addAstVertex(vertex, *expressionVertex);
      }
      break;
    }
    case ConnectingCfg: {
      addCfgVertices(vertex);
      break;
    }
    case ConnectingUses: {
      break;
    }
  }

  BOOST_FOREACH(boost::shared_ptr<ranally::ExpressionVertex>
    expressionVertex, vertex.expressions()) {
    expressionVertex->Accept(*this);
  }
}



void DotVisitor::Visit(
  ScriptVertex& vertex)
{
  _script =
    "digraph G {\n"
    "rank=BT;\n"
    ;

  _mode = Declaring;
  // TODO Use script name.
  _script +=
    UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
    " [label=\"Script\"];\n";
  BOOST_FOREACH(boost::shared_ptr<ranally::StatementVertex> statementVertex,
    vertex.statements()) {
    statementVertex->Accept(*this);
  }

  switch(_type) {
    case Ast: {
      _mode = ConnectingAst;
      BOOST_FOREACH(boost::shared_ptr<ranally::StatementVertex> statementVertex,
        vertex.statements()) {
        addAstVertex(vertex, *statementVertex);
        statementVertex->Accept(*this);
      }

      _mode = ConnectingCfg;
      addCfgVertices(vertex);
      BOOST_FOREACH(boost::shared_ptr<ranally::StatementVertex> statementVertex,
        vertex.statements()) {
        statementVertex->Accept(*this);
      }

      _mode = ConnectingUses;
      BOOST_FOREACH(boost::shared_ptr<ranally::StatementVertex> statementVertex,
        vertex.statements()) {
        statementVertex->Accept(*this);
      }

      break;
    }
    case Flowgraph: {
      _mode = ConnectingFlowgraph;

      BOOST_FOREACH(boost::shared_ptr<ranally::StatementVertex> statementVertex,
        vertex.statements()) {
        statementVertex->Accept(*this);
      }

      break;
    }
  }

  _script += "}\n";
}



void DotVisitor::Visit(
  StringVertex& vertex)
{
  switch(_mode) {
    case Declaring: {
      _script +=
        UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
        " [label=\"\\\"" + vertex.value() + "\\\"\", shape=box];\n";
      break;
    }
    case ConnectingAst: {
      break;
    }
    case ConnectingCfg: {
      addCfgVertices(vertex);
      break;
    }
    case ConnectingUses: {
      break;
    }
  }
}



void DotVisitor::Visit(
  NameVertex& vertex)
{
  switch(_mode) {
    case Declaring: {
      _script +=
        UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
        " [label=\"" + vertex.name() + "\"];\n";
      break;
    }
    case ConnectingAst: {
      break;
    }
    case ConnectingCfg: {
      addCfgVertices(vertex);
      break;
    }
    case ConnectingUses: {
      addUseVertices(vertex);
      break;
    }
  }
}



template<typename T>
void DotVisitor::Visit(
  NumberVertex<T>& vertex)
{
  switch(_mode) {
    case Declaring: {
      _script +=
        UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
        " [label=\"" + (boost::format("%1%") % vertex.value()).str().c_str() +
        "\", shape=box];\n";
      break;
    }
    case ConnectingAst: {
      break;
    }
    case ConnectingCfg: {
      addCfgVertices(vertex);
      break;
    }
    case ConnectingUses: {
      break;
    }
  }
}



void DotVisitor::Visit(
  NumberVertex<int8_t>& vertex)
{
  return Visit<int8_t>(vertex);
}



void DotVisitor::Visit(
  NumberVertex<int16_t>& vertex)
{
  return Visit<int16_t>(vertex);
}



void DotVisitor::Visit(
  NumberVertex<int32_t>& vertex)
{
  return Visit<int32_t>(vertex);
}



void DotVisitor::Visit(
  NumberVertex<int64_t>& vertex)
{
  return Visit<int64_t>(vertex);
}



void DotVisitor::Visit(
  NumberVertex<uint8_t>& vertex)
{
  return Visit<uint8_t>(vertex);
}



void DotVisitor::Visit(
  NumberVertex<uint16_t>& vertex)
{
  return Visit<uint16_t>(vertex);
}



void DotVisitor::Visit(
  NumberVertex<uint32_t>& vertex)
{
  return Visit<uint32_t>(vertex);
}



void DotVisitor::Visit(
  NumberVertex<uint64_t>& vertex)
{
  return Visit<uint64_t>(vertex);
}



void DotVisitor::Visit(
  NumberVertex<float>& vertex)
{
  return Visit<float>(vertex);
}



void DotVisitor::Visit(
  NumberVertex<double>& vertex)
{
  return Visit<double>(vertex);
}



void DotVisitor::Visit(
  IfVertex& vertex)
{
  switch(_mode) {
    case Declaring: {
      _script +=
        UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
        " [label=\"If\", shape=diamond];\n";
      break;
    }
    case ConnectingAst: {
      addAstVertex(vertex, *vertex.condition());
      BOOST_FOREACH(boost::shared_ptr<ranally::StatementVertex>
        statementVertex, vertex.trueStatements()) {
        addAstVertex(vertex, *statementVertex);
      }
      BOOST_FOREACH(boost::shared_ptr<ranally::StatementVertex>
        statementVertex, vertex.falseStatements()) {
        addAstVertex(vertex, *statementVertex);
      }
      break;
    }
    case ConnectingCfg: {
      addCfgVertices(vertex);
      break;
    }
    case ConnectingUses: {
      break;
    }
  }

  vertex.condition()->Accept(*this);
  BOOST_FOREACH(boost::shared_ptr<ranally::StatementVertex>
    statementVertex, vertex.trueStatements()) {
    statementVertex->Accept(*this);
  }
  BOOST_FOREACH(boost::shared_ptr<ranally::StatementVertex>
    statementVertex, vertex.falseStatements()) {
    statementVertex->Accept(*this);
  }
}



void DotVisitor::Visit(
  WhileVertex& vertex)
{
  switch(_mode) {
    case Declaring: {
      _script +=
        UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
        " [label=\"While\", shape=diamond];\n";
      break;
    }
    case ConnectingAst: {
      addAstVertex(vertex, *vertex.condition());
      BOOST_FOREACH(boost::shared_ptr<ranally::StatementVertex>
        statementVertex, vertex.trueStatements()) {
        addAstVertex(vertex, *statementVertex);
      }
      BOOST_FOREACH(boost::shared_ptr<ranally::StatementVertex>
        statementVertex, vertex.falseStatements()) {
        addAstVertex(vertex, *statementVertex);
      }
      break;
    }
    case ConnectingCfg: {
      addCfgVertices(vertex);
      break;
    }
    case ConnectingUses: {
      break;
    }
  }

  vertex.condition()->Accept(*this);
  BOOST_FOREACH(boost::shared_ptr<ranally::StatementVertex>
    statementVertex, vertex.trueStatements()) {
    statementVertex->Accept(*this);
  }
  BOOST_FOREACH(boost::shared_ptr<ranally::StatementVertex>
    statementVertex, vertex.falseStatements()) {
    statementVertex->Accept(*this);
  }
}

} // namespace ranally

