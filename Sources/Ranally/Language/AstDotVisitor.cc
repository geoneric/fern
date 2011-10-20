#include "Ranally/Language/AstDotVisitor.h"
#include <boost/foreach.hpp>
#include <boost/format.hpp>
#include "dev_UnicodeUtils.h"
#include "Ranally/Language/Vertices.h"



namespace ranally {

AstDotVisitor::AstDotVisitor(
  int modes)

  : DotVisitor(),
    _mode(Declaring),
    _modes(modes)

{
}



AstDotVisitor::~AstDotVisitor()
{
}



void AstDotVisitor::setMode(
  Mode mode)
{
  _mode = mode;
}



void AstDotVisitor::addAstVertex(
  language::SyntaxVertex const& sourceVertex,
  language::SyntaxVertex const& targetVertex)
{
  assert(_mode == ConnectingAst);
  addScript(
    UnicodeString((boost::format("\"%1%\"") % &sourceVertex).str().c_str()) +
    " -> " +
    (boost::format("\"%1%\"") % &targetVertex).str().c_str() + " ["
    "];\n"
  );
}



void AstDotVisitor::addCfgVertices(
  language::SyntaxVertex const& sourceVertex)
{
  assert(_mode == ConnectingCfg);
  BOOST_FOREACH(language::SyntaxVertex const* successor,
    sourceVertex.successors()) {
    addScript(
      UnicodeString((boost::format("\"%1%\"") % &sourceVertex).str().c_str()) +
      " -> " +
      (boost::format("\"%1%\"") % successor).str().c_str() + " ["
        "color=red, "
        // TODO Doesn't work when contraint is false. Bug in Dot.
        "constraint=true, "
        "style=dashed"
      "];\n"
    );
  }
}



void AstDotVisitor::addUseVertices(
  language::NameVertex const& vertex)
{
  assert(_mode == ConnectingUses);
  BOOST_FOREACH(language::NameVertex const* use, vertex.uses()) {
    addScript(
      UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
      " -> " +
      (boost::format("\"%1%\"") % use).str().c_str() + " ["
        "color=blue, "
        // TODO Doesn't work when contraint is false. Bug in Dot.
        "constraint=true, "
        "style=dotted"
      "];\n"
    );
  }
}



template<typename T>
void AstDotVisitor::Visit(
  language::NumberVertex<T>& vertex)
{
  switch(_mode) {
    case Declaring: {
      addScript(
        UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
        " [label=\"" + (boost::format("%1%") % vertex.value()).str().c_str() +
        "\", shape=box];\n"
      );
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



void AstDotVisitor::Visit(
  language::NumberVertex<int8_t>& vertex)
{
  return Visit<int8_t>(vertex);
}



void AstDotVisitor::Visit(
  language::NumberVertex<int16_t>& vertex)
{
  return Visit<int16_t>(vertex);
}



void AstDotVisitor::Visit(
  language::NumberVertex<int32_t>& vertex)
{
  return Visit<int32_t>(vertex);
}



void AstDotVisitor::Visit(
  language::NumberVertex<int64_t>& vertex)
{
  return Visit<int64_t>(vertex);
}



void AstDotVisitor::Visit(
  language::NumberVertex<uint8_t>& vertex)
{
  return Visit<uint8_t>(vertex);
}



void AstDotVisitor::Visit(
  language::NumberVertex<uint16_t>& vertex)
{
  return Visit<uint16_t>(vertex);
}



void AstDotVisitor::Visit(
  language::NumberVertex<uint32_t>& vertex)
{
  return Visit<uint32_t>(vertex);
}



void AstDotVisitor::Visit(
  language::NumberVertex<uint64_t>& vertex)
{
  return Visit<uint64_t>(vertex);
}



void AstDotVisitor::Visit(
  language::NumberVertex<float>& vertex)
{
  return Visit<float>(vertex);
}



void AstDotVisitor::Visit(
  language::NumberVertex<double>& vertex)
{
  return Visit<double>(vertex);
}



void AstDotVisitor::Visit(
  language::AssignmentVertex& vertex)
{
  language::ExpressionVertices const& targets = vertex.targets();
  language::ExpressionVertices const& expressions = vertex.expressions();

  switch(_mode) {
    case Declaring: {
      addScript(
        UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
        " [label=\"=\"];\n");
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
  }

  BOOST_FOREACH(boost::shared_ptr<language::ExpressionVertex>
    expressionVertex, vertex.expressions()) {
    expressionVertex->Accept(*this);
  }

  BOOST_FOREACH(boost::shared_ptr<language::ExpressionVertex>
    expressionVertex, vertex.targets()) {
    expressionVertex->Accept(*this);
  }
}



void AstDotVisitor::Visit(
  language::OperatorVertex& vertex)
{
  switch(_mode) {
    case Declaring: {
      // TODO Implement symbol member.
      addScript(
        UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
        " [label=\"" + vertex.symbol() + "\"];\n"
      );
      break;
    }
    case ConnectingAst: {
      BOOST_FOREACH(boost::shared_ptr<language::ExpressionVertex>
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

  BOOST_FOREACH(boost::shared_ptr<language::ExpressionVertex>
    expressionVertex, vertex.expressions()) {
    expressionVertex->Accept(*this);
  }
}



void AstDotVisitor::Visit(
  language::FunctionVertex& vertex)
{
  switch(_mode) {
    case Declaring: {
      addScript(
        UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
        " [label=\"" + vertex.name() + "\"];\n");
      break;
    }
    case ConnectingAst: {
      BOOST_FOREACH(boost::shared_ptr<language::ExpressionVertex>
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

  BOOST_FOREACH(boost::shared_ptr<language::ExpressionVertex>
    expressionVertex, vertex.expressions()) {
    expressionVertex->Accept(*this);
  }
}



void AstDotVisitor::Visit(
  language::IfVertex& vertex)
{
  switch(_mode) {
    case Declaring: {
      addScript(
        UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
        " [label=\"If\", shape=diamond];\n"
      );
      break;
    }
    case ConnectingAst: {
      addAstVertex(vertex, *vertex.condition());
      BOOST_FOREACH(boost::shared_ptr<language::StatementVertex>
        statementVertex, vertex.trueStatements()) {
        addAstVertex(vertex, *statementVertex);
      }
      BOOST_FOREACH(boost::shared_ptr<language::StatementVertex>
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
  BOOST_FOREACH(boost::shared_ptr<language::StatementVertex>
    statementVertex, vertex.trueStatements()) {
    statementVertex->Accept(*this);
  }
  BOOST_FOREACH(boost::shared_ptr<language::StatementVertex>
    statementVertex, vertex.falseStatements()) {
    statementVertex->Accept(*this);
  }
}



void AstDotVisitor::Visit(
  language::NameVertex& vertex)
{
  switch(_mode) {
    case Declaring: {
      addScript(
        UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
        " [label=\"" + vertex.name() + "\"];\n"
      );
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



void AstDotVisitor::Visit(
  language::ScriptVertex& vertex)
{
  setScript(UnicodeString(
    "digraph G {\n"
    "ordering=out;\n"
    "rankdir=TB;\n"
  ));

  setMode(Declaring);
  addScript(
    UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
    (boost::format(" [label=\"%1%\"];\n")
      % dev::encodeInUTF8(vertex.sourceName())).str().c_str());

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

  if(_modes & ConnectingCfg) {
    setMode(ConnectingCfg);
    addCfgVertices(vertex);
    BOOST_FOREACH(boost::shared_ptr<language::StatementVertex>
      statementVertex, vertex.statements()) {
      statementVertex->Accept(*this);
    }
  }

  if(_modes & ConnectingUses) {
    setMode(ConnectingUses);
    BOOST_FOREACH(boost::shared_ptr<language::StatementVertex>
      statementVertex, vertex.statements()) {
      statementVertex->Accept(*this);
    }
  }

  addScript("}\n");
}



void AstDotVisitor::Visit(
  language::StringVertex& vertex)
{
  switch(_mode) {
    case Declaring: {
      addScript(
        UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
        " [label=\"\\\"" + vertex.value() + "\\\"\", shape=box];\n"
      );
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



void AstDotVisitor::Visit(
  language::WhileVertex& vertex)
{
  switch(_mode) {
    case Declaring: {
      addScript(
        UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
        " [label=\"While\", shape=diamond];\n"
      );
      break;
    }
    case ConnectingAst: {
      addAstVertex(vertex, *vertex.condition());
      BOOST_FOREACH(boost::shared_ptr<language::StatementVertex>
        statementVertex, vertex.trueStatements()) {
        addAstVertex(vertex, *statementVertex);
      }
      BOOST_FOREACH(boost::shared_ptr<language::StatementVertex>
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
  BOOST_FOREACH(boost::shared_ptr<language::StatementVertex>
    statementVertex, vertex.trueStatements()) {
    statementVertex->Accept(*this);
  }
  BOOST_FOREACH(boost::shared_ptr<language::StatementVertex>
    statementVertex, vertex.falseStatements()) {
    statementVertex->Accept(*this);
  }
}

} // namespace ranally

