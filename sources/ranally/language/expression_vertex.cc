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
    int lineNr,
    int colId,
    String const& name)

    : StatementVertex(lineNr, colId),
      _name(name)

{
}


String const& ExpressionVertex::name() const
{
    return _name;
}


// void ExpressionVertex::setDataType(
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
// void ExpressionVertex::setValueType(
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


void ExpressionVertex::setResultTypes(
    std::vector<ResultType> const& resultTypes)
{
    _resultTypes = resultTypes;
}


void ExpressionVertex::addResultType(
    DataType data_type,
    ValueType value_type)
{
    _resultTypes.push_back(std::make_tuple(data_type, value_type));
}


std::vector<ExpressionVertex::ResultType> const&
ExpressionVertex::resultTypes() const
{
    return _resultTypes;
}


DataType ExpressionVertex::data_type(
  size_t index) const
{
    assert(index < _resultTypes.size());
    return std::get<0>(_resultTypes[index]);
}


ValueType ExpressionVertex::value_type(
    size_t index) const
{
    assert(index < _resultTypes.size());
    return std::get<1>(_resultTypes[index]);
}


void ExpressionVertex::setValue(
    ExpressionVertexPtr const& value)
{
    _value = value;
}


ExpressionVertexPtr const& ExpressionVertex::value() const
{
    return _value;
}

} // namespace ranally
