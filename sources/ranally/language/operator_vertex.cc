#include "ranally/language/operator_vertex.h"
#include <map>
#include <stdexcept>
#include "ranally/core/string.h"


namespace ranally {

String name_to_symbol(
    String const& name,
    size_t nr_operands)
{
    std::map<String, String> symbols;

    if(nr_operands == 1) {
        // Unary operators.
        symbols["Invert"] = "~";
        symbols["Not"] = "!";
        symbols["Add"] = "+";
        symbols["Sub"] = "-";
    }
    else if(nr_operands == 2) {
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
            % name.encode_in_utf8() % nr_operands).str().c_str());
    }

    return symbols[name];
}


OperatorVertex::OperatorVertex(
    String const& name,
    ExpressionVertices const& expressions)

    : OperationVertex(name, expressions),
      _symbol(name_to_symbol(name, expressions.size()))

{
}


String const& OperatorVertex::symbol() const
{
    return _symbol;
}

} // namespace ranally
