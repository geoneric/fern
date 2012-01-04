#include "Ranally/Language/OperatorVertex.h"
#include <stdexcept>
#include <map>
#include <boost/format.hpp>
#include "Ranally/Util/String.h"



namespace ranally {
namespace language {

UnicodeString nameToSymbol(
  UnicodeString const& name,
  size_t nrOperands)
{
  std::map<UnicodeString, UnicodeString> symbols;

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
      % util::encodeInUTF8(name) % nrOperands).str().c_str());
  }

  return symbols[name];
}



OperatorVertex::OperatorVertex(
  UnicodeString const& name,
  ExpressionVertices const& expressions)

  : OperationVertex(name, expressions),
    _symbol(nameToSymbol(name, expressions.size()))

{
}



OperatorVertex::~OperatorVertex()
{
}



UnicodeString const& OperatorVertex::symbol() const
{
  return _symbol;
}

} // namespace language
} // namespace ranally

