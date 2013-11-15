#include "fern/ast/core/expression_vertex.h"


namespace fern {

ExpressionVertex::ExpressionVertex(
    String const& name)

    : StatementVertex(),
      _name(name)

{
}


ExpressionVertex::ExpressionVertex(
    int line_nr,
    int col_id,
    String const& name)

    : StatementVertex(line_nr, col_id),
      _name(name)

{
}


String const& ExpressionVertex::name() const
{
    return _name;
}


void ExpressionVertex::set_expression_types(
    ExpressionTypes const& expression_types)
{
    _expression_types = expression_types;
}


void ExpressionVertex::add_result_type(
    ExpressionType const& expression_type)
{
    _expression_types.add(expression_type);
}


ExpressionTypes const& ExpressionVertex::expression_types() const
{
    return _expression_types;
}


void ExpressionVertex::set_value(
    ExpressionVertexPtr const& value)
{
    _value = value;
}


ExpressionVertexPtr const& ExpressionVertex::value() const
{
    return _value;
}

} // namespace fern
