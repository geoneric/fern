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
    _operations.clear();
  }

  void Operations(
    OperationsData const& operations)
  {
    assert(_operations.empty());
    _operations = operations;
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
  };

  std::stack<OperationData> _dataStack;

public:

  void pre()
  {
    _dataStack.push(OperationData());
  }

  void Name(
    std::string const& name)
  {
    assert(!_dataStack.empty());
    _dataStack.top().name = dev::decodeFromUTF8(name);
  }

  ranally::operation::OperationPtr post_Operation()
  {
    assert(!_dataStack.empty());
    OperationData result(_dataStack.top());
    _dataStack.pop();
    return boost::make_shared<ranally::operation::Operation>(result.name);
  }

};



class DataType_pimpl:
  public ranally::operation::DataType_pskel
{

private:

  std::string      _dataType;

  static ranally::operation::DataType stringToDataType(
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

public:

  void DataType(
    std::string const& dataType)
  {
    _dataType = dataType;
  }

  std::string post_string()
  {
    return _dataType;
  }

  ranally::operation::DataType post_DataType()
  {
    assert(_dataType.empty());
    return stringToDataType(_dataType);
  }

};



class ValueType_pimpl:
  public ranally::operation::ValueType_pskel
{

private:

  std::string      _dataType;

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

public:

  void ValueType(
    std::string const& dataType)
  {
    _dataType = dataType;
  }

  std::string post_string()
  {
    return _dataType;
  }

  ranally::operation::ValueType post_ValueType()
  {
    assert(_dataType.empty());
    return stringToValueType(_dataType);
  }

};

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
  DataType_pimpl dataType_p;
  ValueType_pimpl valueType_p;

  // DataTypes_pimpl dataTypes_p;
  // dataTypes_p.parsers(dataType_p);

  // ValueTypes_pimpl valueTypes_p;
  // valueTypes_p.parsers(valueType_p);

  // Parameter_pimpl parameter_p;
  // parameter_p.parsers(string_p, string_p, dataTypes_p, valueTypes_p);

  // Parameters_pimpl parameters_p;
  // parameters_p.parsers(parameter_p);

  // Result_pimpl result_p;
  // result_p.parsers(string_p, string_p, dataType_p, valueType_p);

  // Results_pimpl results_p;
  // results_p.parsers(result_p);

  Operation_pimpl operation_p;
  // operation_p.parsers(string_p, string_p, parameters_p, results_p);

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

