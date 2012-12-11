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
    std::vector<ResultType> const& result_types)
{
    _result_types = result_types;
}


void ExpressionVertex::add_result_type(
    DataTypes data_type,
    ValueTypes value_type)
{
    _result_types.push_back(std::make_tuple(data_type, value_type));
}


std::vector<ExpressionVertex::ResultType> const&
ExpressionVertex::result_types() const
{
    return _result_types;
}


DataTypes ExpressionVertex::data_type(
  size_t index) const
{
    assert(index < _result_types.size());
    return std::get<0>(_result_types[index]);
}


ValueTypes ExpressionVertex::value_type(
    size_t index) const
{
    assert(index < _result_types.size());
    return std::get<1>(_result_types[index]);
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
