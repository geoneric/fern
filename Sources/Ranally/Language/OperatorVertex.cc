#include "Ranally/Language/OperatorVertex.h"
#include <stdexcept>
#include <map>
#include "Ranally/Util/String.h"


namespace ranally {

String nameToSymbol(
    String const& name,
    size_t nrOperands)
{
    std::map<String, String> symbols;

    if(nrOperands == 1) {
        // Unary operators.
        symbols["Invert"] = "~";
        symbols["Not"] = "!";
        symbols["Add"] = "+";
        symbols["Sub"] = "-";
    }
    else if(nrOperands == 2) {
        // Binary operators.
        symbols["Add"] = "+";
        symbols["Sub"] = "-";
        symbols["Mult"] = "*";
        symbols["Div"] = "/";
        symbols["Mod"] = "%";
        symbols["Pow"] = "**";
        symbols["LShift"] = "<<";
        symbols["RShift"] = ">>";
        symbols["BitOr"] = "|";
        symbols["BitXor"] = "^";
        symbols["BitAnd"] = "&";
        symbols["FloorDiv"] = "//";

        // Boolean operators.
        symbols["And"] = "and";
        symbols["Or"] = "or";

        // Comparison operators.
        symbols["Eq"] = "==";
        symbols["NotEq"] = "!=";
        symbols["Lt"] = "<";
        symbols["LtE"] = "<=";
        symbols["Gt"] = ">";
        symbols["GtE"] = ">=";
    }

    if(symbols.find(name) == symbols.end()) {
        throw std::runtime_error((boost::format(
            "operator %1% with %2% operands not available")
            % name.encodeInUTF8() % nrOperands).str().c_str());
    }

    return symbols[name];
}


OperatorVertex::OperatorVertex(
    String const& name,
    language::ExpressionVertices const& expressions)

    : OperationVertex(name, expressions),
      _symbol(nameToSymbol(name, expressions.size()))

{
}


OperatorVertex::~OperatorVertex()
{
}


String const& OperatorVertex::symbol() const
{
    return _symbol;
}

} // namespace ranally
