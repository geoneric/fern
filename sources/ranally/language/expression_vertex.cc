#include "ranally/language/expression_vertex.h"


namespace ranally {

ExpressionVertex::ExpressionVertex(
    String const& name)

    : StatementVertex(),
      _name(name) // ,
      // _data_type(DataType::DT_UNKNOWN),
      // _value_type(VT_UNKNOWN)

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


// void ExpressionVertex::set_data_type(
//   DataType data_type)
// {
//   _data_type = data_type;
// }
// 
// 
// 
// DataType ExpressionVertex::data_type() const
// {
//   return _data_type;
// }
// 
// 
// 
// void ExpressionVertex::set_value_type(
//   ValueType value_type)
// {
//   _value_type = value_type;
// }
// 
// 
// 
// ValueType ExpressionVertex::value_type() const
// {
//   return _value_type;
// }


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


DataTypes ExpressionVertex::data_type(
  size_t index) const
{
    assert(index < _result_types.size());
    return _result_types[index].data_type();
}


ValueTypes ExpressionVertex::value_type(
    size_t index) const
{
    assert(index < _result_types.size());
    return _result_types[index].value_type();
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

} // namespace ranally
