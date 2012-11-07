#include "Ranally/Language/ExpressionVertex.h"


namespace ranally {

ExpressionVertex::ExpressionVertex(
    String const& name)

    : StatementVertex(),
      _name(name) // ,
      // _dataType(DT_UNKNOWN),
      // _valueType(VT_UNKNOWN)

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
//   DataType dataType)
// {
//   _dataType = dataType;
// }
// 
// 
// 
// DataType ExpressionVertex::dataType() const
// {
//   return _dataType;
// }
// 
// 
// 
// void ExpressionVertex::setValueType(
//   ValueType valueType)
// {
//   _valueType = valueType;
// }
// 
// 
// 
// ValueType ExpressionVertex::valueType() const
// {
//   return _valueType;
// }


void ExpressionVertex::setResultTypes(
    std::vector<ResultType> const& resultTypes)
{
    _resultTypes = resultTypes;
}


void ExpressionVertex::addResultType(
    DataType dataType,
    ValueType valueType)
{
    _resultTypes.push_back(boost::make_tuple(dataType, valueType));
}


std::vector<ExpressionVertex::ResultType> const&
ExpressionVertex::resultTypes() const
{
    return _resultTypes;
}


DataType ExpressionVertex::dataType(
  size_t index) const
{
    assert(index < _resultTypes.size());
    return _resultTypes[index].get<0>();
}


ValueType ExpressionVertex::valueType(
    size_t index) const
{
    assert(index < _resultTypes.size());
    return _resultTypes[index].get<1>();
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
