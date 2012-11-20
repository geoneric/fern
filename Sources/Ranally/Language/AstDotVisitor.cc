#include "Ranally/Language/AstDotVisitor.h"
#include "Ranally/Util/string.h"
#include "Ranally/Language/Vertices.h"


namespace ranally {
namespace {

String join(
    std::vector<String> const& strings,
    String const& separator)
{
    String result;

    if(!strings.empty()) {
        result += strings.front();

        for(size_t i = 1; i < strings.size(); ++i) {
            result += separator + strings[i];
        }
    }

    return result;
}


String dataTypesToString(
    DataTypes const& dataTypes)
{
    std::vector<String> strings;

    if(dataTypes & DataType::DT_UNKNOWN) {
        strings.push_back("?");
    }
    else {
        if(dataTypes & DataType::DT_VALUE) {
            strings.push_back("val");
        }
        if(dataTypes & DataType::DT_RASTER) {
            strings.push_back("rst");
        }
        if(dataTypes & DataType::DT_FEATURE) {
            strings.push_back("ftr");
        }
        if(dataTypes & DataType::DT_DEPENDS_ON_INPUT) {
            strings.push_back("dep");
        }
    }

    assert(!strings.empty());
    return join(strings, "|");
}


String valueTypesToString(
    ValueTypes const& valueTypes)
{
    std::vector<String> strings;

    if(valueTypes & DataType::DT_UNKNOWN) {
        strings.push_back("?");
    }
    else {
        if(valueTypes & VT_UINT8) {
            strings.push_back("u8");
        }
        if(valueTypes & VT_INT8) {
            strings.push_back("s8");
        }
        if(valueTypes & VT_UINT16) {
            strings.push_back("u16");
        }
        if(valueTypes & VT_INT16) {
            strings.push_back("s16");
        }
        if(valueTypes & VT_UINT32) {
            strings.push_back("u32");
        }
        if(valueTypes & VT_INT32) {
            strings.push_back("s32");
        }
        if(valueTypes & VT_UINT64) {
            strings.push_back("u64");
        }
        if(valueTypes & VT_INT64) {
            strings.push_back("s64");
        }
        if(valueTypes & VT_FLOAT32) {
            strings.push_back("f32");
        }
        if(valueTypes & VT_FLOAT64) {
            strings.push_back("f64");
        }
        if(valueTypes & VT_STRING) {
            strings.push_back("str");
        }
        if(valueTypes & VT_DEPENDS_ON_INPUT) {
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
      _mode(Mode::Declaring),
      _modes(modes)

{
}


void AstDotVisitor::setMode(
    Mode mode)
{
    _mode = mode;
}


void AstDotVisitor::addAstVertex(
    SyntaxVertex const& sourceVertex,
    SyntaxVertex const& targetVertex)
{
    assert(_mode == Mode::ConnectingAst);
    addScript(
        String(boost::format("\"%1%\"") % &sourceVertex) + " -> " +
        String(boost::format("\"%1%\"") % &targetVertex) + " ["
        "];\n"
    );
}


void AstDotVisitor::addCfgVertices(
    SyntaxVertex const& sourceVertex)
{
    assert(_mode == Mode::ConnectingCfg);
    for(auto successor: sourceVertex.successors()) {
        addScript(
            String(boost::format("\"%1%\"") % &sourceVertex) + " -> " +
            String(boost::format("\"%1%\"") % successor) + " ["
                "color=\"/spectral9/2\", "
                "constraint=false, "
                "style=dashed, "
                "penwidth=0.25"
            "];\n"
        );
    }
}


void AstDotVisitor::addUseVertices(
    NameVertex const& vertex)
{
    assert(_mode == Mode::ConnectingUses);
    for(auto use: vertex.uses()) {
        addScript(
            String(boost::format("\"%1%\"") % &vertex) + " -> " +
            String(boost::format("\"%1%\"") % use) + " ["
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
        case Mode::ConnectingAst: {
            break;
        }
        case Mode::ConnectingCfg: {
            addCfgVertices(vertex);
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
    NumberVertex<type>& vertex)                                      \
{                                                                              \
    Visit<type>(vertex);                                                       \
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
    AssignmentVertex& vertex)
{
    switch(_mode) {
        case Mode::Declaring: {
            addScript(
                String(boost::format("\"%1%\"") % &vertex) +
                " [label=\"=\"];\n");
            break;
        }
        case Mode::ConnectingAst: {
            addAstVertex(vertex, *vertex.target());
            addAstVertex(vertex, *vertex.expression());
            break;
        }
        case Mode::ConnectingCfg: {
            addCfgVertices(vertex);
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
            addScript(
                String(boost::format("\"%1%\"") % &vertex) +
                " [label=\"" + vertex.symbol() + "\"];\n"
            );
            break;
        }
        case Mode::ConnectingAst: {
            for(auto expressionVertex: vertex.expressions()) {
                addAstVertex(vertex, *expressionVertex);
            }
            break;
        }
        case Mode::ConnectingCfg: {
            addCfgVertices(vertex);
            break;
        }
        case Mode::ConnectingUses: {
            break;
        }
    }

    for(auto expressionVertex: vertex.expressions()) {
        expressionVertex->Accept(*this);
    }
}


void AstDotVisitor::Visit(
    FunctionVertex& vertex)
{
    switch(_mode) {
        case Mode::Declaring: {
            addScript(
                String(boost::format("\"%1%\"") % &vertex) +
                " [label=\"" + vertex.name() + "\"];\n");
            break;
        }
        case Mode::ConnectingAst: {
            for(auto expressionVertex: vertex.expressions()) {
                addAstVertex(vertex, *expressionVertex);
            }
            break;
        }
        case Mode::ConnectingCfg: {
            addCfgVertices(vertex);
            break;
        }
        case Mode::ConnectingUses: {
            break;
        }
    }

    for(auto expressionVertex: vertex.expressions()) {
        expressionVertex->Accept(*this);
    }
}


void AstDotVisitor::Visit(
    IfVertex& vertex)
{
    switch(_mode) {
        case Mode::Declaring: {
            addScript(
                String(boost::format("\"%1%\"") % &vertex) +
                " [label=\"If\", shape=diamond];\n"
            );
            break;
        }
        case Mode::ConnectingAst: {
            addAstVertex(vertex, *vertex.condition());
            for(auto statementVertex: vertex.trueStatements()) {
                addAstVertex(vertex, *statementVertex);
            }
            for(auto statementVertex: vertex.falseStatements()) {
                addAstVertex(vertex, *statementVertex);
            }
            break;
        }
        case Mode::ConnectingCfg: {
            addCfgVertices(vertex);
            break;
        }
        case Mode::ConnectingUses: {
            break;
        }
    }

    vertex.condition()->Accept(*this);
    for(auto statementVertex: vertex.trueStatements()) {
        statementVertex->Accept(*this);
    }
    for(auto statementVertex: vertex.falseStatements()) {
        statementVertex->Accept(*this);
    }
}


void AstDotVisitor::Visit(
    NameVertex& vertex)
{
    switch(_mode) {
        case Mode::Declaring: {
            std::vector<String> attributes;
            String label = vertex.name();

            std::vector<ExpressionVertex::ResultType> const&
                resultTypes(vertex.resultTypes());
            if(!resultTypes.empty()) {
                assert(resultTypes.size() == 1);
                String dataTypes = dataTypesToString(std::get<0>(
                    resultTypes[0]));
                String valueTypes = valueTypesToString(std::get<1>(
                    resultTypes[0]));

                label += String("\\n") +
                    "dt: " + dataTypes + "\\n" +
                    "vt: " + valueTypes;
            }

            attributes.push_back("label=\"" + label + "\"");

            addScript(
                String(boost::format("\"%1%\"") % &vertex) + " [" +
                join(attributes, ", ") + "];\n"
            );

            break;
        }
        case Mode::ConnectingAst: {
            break;
        }
        case Mode::ConnectingCfg: {
            addCfgVertices(vertex);
            break;
        }
        case Mode::ConnectingUses: {
            addUseVertices(vertex);
            break;
        }
    }
}


void AstDotVisitor::Visit(
    ScriptVertex& vertex)
{
    // TODO 'ordering=out' is current not supported in combination with
    // TODO 'constraint=false'. Check again with dot > 2.28.0, when it becomes
    // TODO available.
    setScript(String(
        "digraph G {\n"
        "// ordering=out;\n"
        "rankdir=TB;\n"
    ));

    setMode(Mode::Declaring);
    addScript(
        String(boost::format("\"%1%\"") % &vertex) +
        String(boost::format(" [label=\"%1%\"];\n")
            % vertex.sourceName().encodeInUTF8())
    );

    for(auto statementVertex: vertex.statements()) {
        statementVertex->Accept(*this);
    }

    setMode(Mode::ConnectingAst);
    for(auto statementVertex: vertex.statements()) {
        addAstVertex(vertex, *statementVertex);
        statementVertex->Accept(*this);
    }

    if(_modes & Mode::ConnectingCfg) {
        setMode(Mode::ConnectingCfg);
        addCfgVertices(vertex);
        for(auto statementVertex: vertex.statements()) {
          statementVertex->Accept(*this);
        }
    }

    if(_modes & Mode::ConnectingUses) {
        setMode(Mode::ConnectingUses);
        for(auto statementVertex: vertex.statements()) {
            statementVertex->Accept(*this);
        }
    }

    addScript("}\n");
}


void AstDotVisitor::Visit(
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
        case Mode::ConnectingAst: {
            break;
        }
        case Mode::ConnectingCfg: {
            addCfgVertices(vertex);
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
            addScript(
                String(boost::format("\"%1%\"") % &vertex) +
                " [label=\"While\", shape=diamond];\n"
            );
            break;
        }
        case Mode::ConnectingAst: {
            addAstVertex(vertex, *vertex.condition());
            for(auto statementVertex: vertex.trueStatements()) {
                addAstVertex(vertex, *statementVertex);
            }
            for(auto statementVertex: vertex.falseStatements()) {
                addAstVertex(vertex, *statementVertex);
            }
            break;
        }
        case Mode::ConnectingCfg: {
            addCfgVertices(vertex);
            break;
        }
        case Mode::ConnectingUses: {
            break;
        }
    }

    vertex.condition()->Accept(*this);
    for(auto statementVertex: vertex.trueStatements()) {
        statementVertex->Accept(*this);
    }
    for(auto statementVertex: vertex.falseStatements()) {
        statementVertex->Accept(*this);
    }
}

} // namespace ranally
