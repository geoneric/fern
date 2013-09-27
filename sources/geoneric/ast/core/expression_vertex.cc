#include "geoneric/ast/core/expression_vertex.h"


namespace geoneric {

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


void ExpressionVertex::set_result_types(
    ResultTypes const& result_types)
{
    _result_types = result_types;
}


void ExpressionVertex::add_result_type(
    ResultType const& result_type)
{
    _result_types.push_back(result_type);
}


ResultTypes const& ExpressionVertex::result_types() const
{
    return _result_types;
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

} // namespace geoneric
