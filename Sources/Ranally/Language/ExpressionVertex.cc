#include "Ranally/Language/ExpressionVertex.h"



namespace ranally {
namespace language {

ExpressionVertex::ExpressionVertex(
  UnicodeString const& name)

  : StatementVertex(),
    _name(name)

{
}



ExpressionVertex::ExpressionVertex(
  int lineNr,
  int colId,
  UnicodeString const& name)

  : StatementVertex(lineNr, colId),
    _name(name)

{
}



ExpressionVertex::~ExpressionVertex()
{
}



UnicodeString const& ExpressionVertex::name() const
{
  return _name;
}



void ExpressionVertex::setDataType(
  operation::DataType dataType)
{
  _dataType = dataType;
}



operation::DataType ExpressionVertex::dataType() const
{
  return _dataType;
}



void ExpressionVertex::setValueType(
  operation::ValueType valueType)
{
  _valueType = valueType;
}



operation::ValueType ExpressionVertex::valueType() const
{
  return _valueType;
}

} // namespace language
} // namespace ranally

