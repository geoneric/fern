#include "Ranally/Language/AstDotVisitor.h"
#include <boost/foreach.hpp>
#include <boost/format.hpp>
#include "Ranally/Util/String.h"
#include "Ranally/Language/Vertices.h"



namespace ranally {
namespace {

UnicodeString join(
  std::vector<UnicodeString> const& strings,
  UnicodeString const& separator)
{
  UnicodeString result;

  if(!strings.empty()) {
    result += strings.front();

    for(size_t i = 1; i < strings.size(); ++i) {
      result += separator + strings[i];
    }
  }

  return result;
}



UnicodeString dataTypesToString(
  operation::DataTypes const& dataTypes)
{
  std::vector<UnicodeString> strings;

  if(dataTypes & operation::DT_UNKNOWN) {
    strings.push_back("?");
  }
  else {
    if(dataTypes & operation::DT_VALUE) {
      strings.push_back("val");
    }
    if(dataTypes & operation::DT_RASTER) {
      strings.push_back("rst");
    }
    if(dataTypes & operation::DT_FEATURE) {
      strings.push_back("ftr");
    }
    if(dataTypes & operation::DT_DEPENDS_ON_INPUT) {
      strings.push_back("dep");
    }
  }

  assert(!strings.empty());
  return join(strings, "|");
}



UnicodeString valueTypesToString(
  operation::ValueTypes const& valueTypes)
{
  std::vector<UnicodeString> strings;

  if(valueTypes & operation::DT_UNKNOWN) {
    strings.push_back("?");
  }
  else {
    if(valueTypes & operation::VT_UINT8) {
      strings.push_back("u8");
    }
    if(valueTypes & operation::VT_INT8) {
      strings.push_back("s8");
    }
    if(valueTypes & operation::VT_UINT16) {
      strings.push_back("u16");
    }
    if(valueTypes & operation::VT_INT16) {
      strings.push_back("s16");
    }
    if(valueTypes & operation::VT_UINT32) {
      strings.push_back("u32");
    }
    if(valueTypes & operation::VT_INT32) {
      strings.push_back("s32");
    }
    if(valueTypes & operation::VT_UINT64) {
      strings.push_back("u64");
    }
    if(valueTypes & operation::VT_INT64) {
      strings.push_back("s64");
    }
    if(valueTypes & operation::VT_FLOAT32) {
      strings.push_back("f32");
    }
    if(valueTypes & operation::VT_FLOAT64) {
      strings.push_back("f64");
    }
    if(valueTypes & operation::VT_STRING) {
      strings.push_back("str");
    }
    if(valueTypes & operation::VT_DEPENDS_ON_INPUT) {
      strings.push_back("dep");
    }
  }

  assert(!strings.empty());
  return join(strings, "|");
}

} // Anonymous namespace



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
        "color=\"/spectral9/2\", "
        "constraint=false, "
        "style=dashed, "
        "penwidth=0.25"
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



#define VISIT_NUMBER_VERTEX(                                                   \
  type)                                                                        \
void AstDotVisitor::Visit(                                                     \
  language::NumberVertex<type>& vertex)                                        \
{                                                                              \
  Visit<type>(vertex); \
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
  language::AssignmentVertex& vertex)
{
  switch(_mode) {
    case Declaring: {
      addScript(
        UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
        " [label=\"=\"];\n");
      break;
    }
    case ConnectingAst: {
      addAstVertex(vertex, *vertex.target());
      addAstVertex(vertex, *vertex.expression());
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

  vertex.expression()->Accept(*this);
  vertex.target()->Accept(*this);
}



void AstDotVisitor::Visit(
  language::OperatorVertex& vertex)
{
  switch(_mode) {
    case Declaring: {
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
      std::vector<UnicodeString> attributes;
      UnicodeString label = vertex.name();

      std::vector<language::ExpressionVertex::ResultType> const& resultTypes(
        vertex.resultTypes());
      if(!resultTypes.empty()) {
        assert(resultTypes.size() == 1);
        UnicodeString dataTypes = dataTypesToString(resultTypes[0].get<0>());
        UnicodeString valueTypes = valueTypesToString(resultTypes[0].get<1>());

        label += UnicodeString("\\n") +
          "dt: " + dataTypes + "\\n" +
          "vt: " + valueTypes;
      }

      attributes.push_back("label=\"" + label + "\"");

      addScript(
        UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
        " [" + join(attributes, ", ") + "];\n"
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
  // TODO 'ordering=out' is current not supported in combination with
  // TODO 'constraint=false'. Check again with dot > 2.28.0, when it becomes
  // TODO available.
  setScript(UnicodeString(
    "digraph G {\n"
    "// ordering=out;\n"
    "rankdir=TB;\n"
  ));

  setMode(Declaring);
  addScript(
    UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
    (boost::format(" [label=\"%1%\"];\n")
      % util::encodeInUTF8(vertex.sourceName())).str().c_str());

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
        " [label=\"\\\"" + vertex.value() +
        "\\\"\", fontname=courier, shape=box];\n"
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

