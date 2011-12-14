#include "Ranally/Operation/XmlParser.h"
#include <sstream>
#include <stack>
#include <vector>
#include <boost/make_shared.hpp>
#include "dev_UnicodeUtils.h"
#include "Operation-pskel.hxx"
#include "Ranally/Operation/DataType.h"
#include "Ranally/Operation/Operations.h"



namespace {

ranally::operation::DataType stringToDataType(
  std::string const& string)
{
  assert(!string.empty());
  ranally::operation::DataType dataType = ranally::operation::DT_UNKNOWN;

  if(string == "Number") {
    dataType = ranally::operation::DT_NUMBER;
  }
  else if(string == "String") {
    dataType = ranally::operation::DT_STRING;
  }
  else if(string == "Raster") {
    dataType = ranally::operation::DT_RASTER;
  }
  else if(string == "Feature") {
    dataType = ranally::operation::DT_FEATURE;
  }
  else if(string == "All") {
    dataType = ranally::operation::DT_ALL;
  }

  assert(dataType != ranally::operation::DT_UNKNOWN);
  return dataType;
}



static ranally::operation::ValueType stringToValueType(
  std::string const& string)
{
  assert(!string.empty());
  ranally::operation::ValueType valueType = ranally::operation::VT_UNKNOWN;

  if(string == "UInt8") {
    valueType = ranally::operation::VT_UINT8;
  }
  else if(string == "Int8") {
    valueType = ranally::operation::VT_INT8;
  }
  else if(string == "UInt16") {
    valueType = ranally::operation::VT_UINT16;
  }
  else if(string == "Int16") {
    valueType = ranally::operation::VT_INT16;
  }
  else if(string == "UInt32") {
    valueType = ranally::operation::VT_UINT32;
  }
  else if(string == "Int32") {
    valueType = ranally::operation::VT_INT32;
  }
  else if(string == "UInt64") {
    valueType = ranally::operation::VT_UINT64;
  }
  else if(string == "Int64") {
    valueType = ranally::operation::VT_INT64;
  }
  else if(string == "Float32") {
    valueType = ranally::operation::VT_FLOAT32;
  }
  else if(string == "Float64") {
    valueType = ranally::operation::VT_FLOAT64;
  }
  else if(string == "String") {
    valueType = ranally::operation::VT_STRING;
  }
  else if(string == "UnsignedInteger") {
    valueType = ranally::operation::VT_UNSIGNED_INTEGER;
  }
  else if(string == "SignedInteger") {
    valueType = ranally::operation::VT_SIGNED_INTEGER;
  }
  else if(string == "Integer") {
    valueType = ranally::operation::VT_INTEGER;
  }
  else if(string == "FloatingPoint") {
    valueType = ranally::operation::VT_FLOATING_POINT;
  }
  else if(string == "Number") {
    valueType = ranally::operation::VT_NUMBER;
  }
  else if(string == "All") {
    valueType = ranally::operation::VT_ALL;
  }

  assert(valueType != ranally::operation::VT_UNKNOWN);
  return valueType;
}



class Operations_pimpl:
  public ranally::operation::Operations_pskel
{

private:

  typedef std::vector<boost::shared_ptr<ranally::operation::Operation> >
    OperationsData;

  OperationsData   _operations;

public:

  void pre()
  {
    assert(_operations.empty());
  }

  void Operation(
    ranally::operation::OperationPtr const& operation)
  {
    _operations.push_back(operation);
  }

  ranally::operation::OperationsPtr post_Operations()
  {
    return boost::make_shared<ranally::operation::Operations>(_operations);
  }

};



class Operation_pimpl:
  public ranally::operation::Operation_pskel
{

private:

  struct OperationData
  {
    UnicodeString name;
    UnicodeString description;
    std::vector<ranally::operation::Parameter> parameters;
    std::vector<ranally::operation::Result> results;
  };

  std::stack<OperationData> _dataStack;

public:

  void pre()
  {
    assert(_dataStack.empty());
    _dataStack.push(OperationData());
  }

  void Name(
    std::string const& name)
  {
    assert(!_dataStack.empty());
    _dataStack.top().name = dev::decodeFromUTF8(name);
  }

  void Description(
    std::string const& description)
  {
    assert(!_dataStack.empty());
    _dataStack.top().description = dev::decodeFromUTF8(description);
  }

  void Parameters(
    std::vector<ranally::operation::Parameter> const& parameters)
  {
    assert(!_dataStack.empty());
    _dataStack.top().parameters = parameters;
  }

  void Results(
    std::vector<ranally::operation::Result> const& results)
  {
    assert(!_dataStack.empty());
    _dataStack.top().results = results;
  }

  ranally::operation::OperationPtr post_Operation()
  {
    assert(_dataStack.size() == 1);
    assert(!_dataStack.empty());
    OperationData result(_dataStack.top());
    _dataStack.pop();
    return boost::make_shared<ranally::operation::Operation>(result.name,
      result.description, result.parameters, result.results);
  }

};



class Parameters_pimpl:
  public ranally::operation::Parameters_pskel
{

private:

  std::vector<ranally::operation::Parameter> _parameters;

public:

  void pre()
  {
    _parameters.clear();
  }

  void Parameter(
    ranally::operation::Parameter const& parameter)
  {
    _parameters.push_back(parameter);
  }

  std::vector<ranally::operation::Parameter> const& post_Parameters()
  {
    return _parameters;
  }

};



class Parameter_pimpl:
  public ranally::operation::Parameter_pskel
{

private:

  struct ParameterData
  {
    UnicodeString name;
    UnicodeString description;
    std::vector<ranally::operation::DataType> dataTypes;
    std::vector<ranally::operation::ValueType> valueTypes;
  };

  std::stack<ParameterData> _dataStack;

public:

  void pre()
  {
    assert(_dataStack.empty());
    _dataStack.push(ParameterData());
  }

  void Name(
    std::string const& name)
  {
    assert(!_dataStack.empty());
    _dataStack.top().name = dev::decodeFromUTF8(name);
  }

  void Description(
    std::string const& description)
  {
    assert(!_dataStack.empty());
    _dataStack.top().description = dev::decodeFromUTF8(description);
  }

  void DataTypes(
    std::vector<ranally::operation::DataType> const& dataTypes)
  {
    assert(!_dataStack.empty());
    _dataStack.top().dataTypes = dataTypes;
  }

  void ValueTypes(
    std::vector<ranally::operation::ValueType> const& valueTypes)
  {
    assert(!_dataStack.empty());
    _dataStack.top().valueTypes = valueTypes;
  }

  ranally::operation::Parameter post_Parameter()
  {
    assert(_dataStack.size() == 1);
    assert(!_dataStack.empty());
    ParameterData result(_dataStack.top());
    _dataStack.pop();
    return ranally::operation::Parameter(result.name, result.description,
      result.dataTypes, result.valueTypes);
  }

};



class Results_pimpl:
  public ranally::operation::Results_pskel
{

private:

  std::vector<ranally::operation::Result> _results;

public:

  void pre()
  {
    _results.clear();
  }

  void Result(
    ranally::operation::Result const& result)
  {
    _results.push_back(result);
  }

  std::vector<ranally::operation::Result> const& post_Results()
  {
    return _results;
  }

};



class Result_pimpl:
  public ranally::operation::Result_pskel
{

private:

  struct ResultData
  {
    UnicodeString name;
    UnicodeString description;
    ranally::operation::DataType dataType;
    ranally::operation::ValueType valueType;
  };

  std::stack<ResultData> _dataStack;

public:

  void pre()
  {
    assert(_dataStack.empty());
    _dataStack.push(ResultData());
  }

  void Name(
    std::string const& name)
  {
    assert(!_dataStack.empty());
    _dataStack.top().name = dev::decodeFromUTF8(name);
  }

  void Description(
    std::string const& description)
  {
    assert(!_dataStack.empty());
    _dataStack.top().description = dev::decodeFromUTF8(description);
  }

  void DataType(
    // ranally::operation::DataType const& dataType)
    std::string const& dataType)
  {
    assert(!_dataStack.empty());
    _dataStack.top().dataType = stringToDataType(dataType);
  }

  void ValueType(
    // ranally::operation::ValueType const& valueType)
    std::string const& valueType)
  {
    assert(!_dataStack.empty());
    _dataStack.top().valueType = stringToValueType(valueType);
  }

  ranally::operation::Result post_Result()
  {
    assert(_dataStack.size() == 1);
    assert(!_dataStack.empty());
    ResultData result(_dataStack.top());
    _dataStack.pop();
    return ranally::operation::Result(result.name, result.description,
      result.dataType, result.valueType);
  }

};



class DataTypes_pimpl:
  public ranally::operation::DataTypes_pskel
{

private:

  std::vector<ranally::operation::DataType> _dataTypes;

public:

  void pre()
  {
    _dataTypes.clear();
  }

  // void DataType(
  //   ranally::operation::DataType const& dataType)
  // {
  //   _dataTypes.push_back(dataType);
  // }

  void DataType(
    std::string const& dataType)
  {
    _dataTypes.push_back(stringToDataType(dataType));
  }

  std::vector<ranally::operation::DataType> const& post_DataTypes()
  {
    return _dataTypes;
  }

};



// class DataType_pimpl:
//   public ranally::operation::DataType_pskel
// {
// 
// private:
// 
//   std::string      _dataType;
// 
// public:
// 
//   void DataType(
//     std::string const& dataType)
//   {
//     assert(false);
//     _dataType = dataType;
//   }
// 
//   std::string post_string()
//   {
//     assert(false);
//     return _dataType;
//   }
// 
//   ranally::operation::DataType post_DataType()
//   {
//     assert(false);
//     assert(_dataType.empty());
//     return stringToDataType(_dataType);
//   }
// 
// };



class ValueTypes_pimpl:
  public ranally::operation::ValueTypes_pskel
{

private:

  std::vector<ranally::operation::ValueType> _valueTypes;

public:

  void pre()
  {
    _valueTypes.clear();
  }

  // void ValueType(
  //   ranally::operation::ValueType const& valueType)
  // {
  //   _valueTypes.push_back(valueType);
  // }

  void ValueType(
    std::string const& valueType)
  {
    _valueTypes.push_back(stringToValueType(valueType));
  }

  std::vector<ranally::operation::ValueType> const& post_ValueTypes()
  {
    return _valueTypes;
  }

};



// class ValueType_pimpl:
//   public ranally::operation::ValueType_pskel
// {
// 
// private:
// 
//   std::string      _dataType;
// 
// public:
// 
//   void ValueType(
//     std::string const& dataType)
//   {
//     _dataType = dataType;
//   }
// 
//   std::string post_string()
//   {
//     return _dataType;
//   }
// 
//   ranally::operation::ValueType post_ValueType()
//   {
//     assert(_dataType.empty());
//     return stringToValueType(_dataType);
//   }
// 
// };

} // Anonymous namespace



namespace ranally {
namespace operation {

XmlParser::XmlParser()
{
}



XmlParser::~XmlParser()
{
}



OperationsPtr XmlParser::parse(
  std::istream& stream) const
{
  xml_schema::string_pimpl string_p;

  // DataType_pimpl dataType_p;
  // ValueType_pimpl valueType_p;

  DataTypes_pimpl dataTypes_p;
  dataTypes_p.parsers(string_p /* dataType_p */);

  ValueTypes_pimpl valueTypes_p;
  valueTypes_p.parsers(string_p /* valueType_p */);

  Parameter_pimpl parameter_p;
  parameter_p.parsers(string_p, string_p, dataTypes_p, valueTypes_p);

  Parameters_pimpl parameters_p;
  parameters_p.parsers(parameter_p);

  Result_pimpl result_p;
  result_p.parsers(string_p, string_p, string_p, string_p
    /* dataType_p, valueType_p */);

  Results_pimpl results_p;
  results_p.parsers(result_p);

  Operation_pimpl operation_p;
  operation_p.parsers(string_p, string_p, parameters_p, results_p);

  Operations_pimpl operations_p;
  operations_p.parsers(operation_p);

  xml_schema::document doc_p(operations_p, "Operations");

  operations_p.pre();
  doc_p.parse(stream);
  return operations_p.post_Operations();
}



OperationsPtr XmlParser::parse(
  UnicodeString const& xml) const
{
  // Copy string contents in a string stream and work with that.
  std::stringstream stream;
  stream.exceptions(std::ifstream::badbit | std::ifstream::failbit);
  stream << dev::encodeInUTF8(xml); // << std::endl;

  return parse(stream);
}

} // namespace operation
} // namespace ranally

