#include "fern/ast/core/operator_vertex.h"
#include <map>
#include "fern/core/string.h"


namespace fern {

std::map<String, String> unary_operator_symbols = {
    { "Invert", "~" },
    { "Not", "!" },
    { "UnaryAdd", "+" },
    { "UnarySub", "-" }
};


std::map<String, String> binary_operator_symbols = {
    // Binary operators.
    { "add"     , "+"   },
    { "Sub"     , "-"   },
    { "Mult"    , "*"   },
    { "Div"     , "/"   },
    { "Mod"     , "%"   },
    { "Pow"     , "**"  },
    { "LShift"  , "<<"  },
    { "RShift"  , ">>"  },
    { "BitOr"   , "|"   },
    { "BitXor"  , "^"   },
    { "BitAnd"  , "&"   },
    { "FloorDiv", "//"  },

    // Boolean operators.
    { "And"     , "and" },
    { "Or"      , "or"  },

    // Comparison operators.
    { "Eq"      , "=="  },
    { "NotEq"   , "!="  },
    { "Lt"      , "<"   },
    { "LtE"     , "<="  },
    { "Gt"      , ">"   },
    { "GtE"     , ">="  }
};


std::map<size_t, std::map<String, String>> operator_symbols = {
    { 1, unary_operator_symbols  },
    { 2, binary_operator_symbols }
};


String name_to_symbol(
    String const& name,
    size_t nr_operands)
{
    assert(operator_symbols.find(nr_operands) != operator_symbols.end());
    assert(operator_symbols[nr_operands].find(name) !=
        operator_symbols[nr_operands].end());
    return operator_symbols[nr_operands][name];
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

} // namespace fern
