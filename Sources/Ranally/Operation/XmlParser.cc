#include "Ranally/Operation/XmlParser.h"
#include <sstream>
#include <stack>
#include <vector>
#include <boost/make_shared.hpp>
#include "dev_UnicodeUtils.h"
#include "Operation-pskel.hxx"
#include "Ranally/Operation/Operations.h"



namespace {

class Operations_pimpl:
  public ranally::operation::Operations_pskel
{

private:

  typedef std::vector<boost::shared_ptr<ranally::operation::Operation> >
    OperationsData;

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

  OperationsPtr post_Operations()
  {
    return boost::make_shared<ranally::operation::Operations>(_operations);
  }

private:

  OperationsData   _operations;

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

  OperationPtr post_Operation()
  {
    assert(!_dataStack.empty());
    OperationData result(_dataStack.top());
    _dataStack.pop();
    return boost::make_shared<ranally::operation::Operation>(result.name);
  }

private:

  Operation_      _operation;

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



boost::shared_ptr<Operations> XmlParser::parse(
  std::istream& stream) const
{
  xml_schema::string_pimpl string_p;

  DataType_pimpl dataType_p;
  dataType_p.parsers(string_p);

  ValueType_pimpl valueType_p;
  valueType_p.parsers(string_p);

  DataTypes_pimpl dataTypes_p;
  dataTypes_p.parsers(dataType_p);

  ValueTypes_pimpl valueTypes_p;
  valueTypes_p.parsers(valueType_p);

  Parameter_pimpl parameter_p;
  parameter_p.parsers(string_p, string_p, dataTypes_p, valueTypes_p);

  Parameters_pimpl parameters_p;
  parameters_p.parsers(parameter_p);

  Result_pimpl result_p;
  result_p.parsers(string_p, string_p, dataType_p, valueType_p);

  Results_pimpl results_p;
  results_p.parsers(result_p);

  Operation_pimpl operation_p;
  operation_p.parsers(string_p, string_p, parameters_p, results_p);

  Operations_pimpl operations_p;
  operations_p.parsers(operation_p);

  xml_schema::document doc_p(operations_p, "Operations");

  operations_p.pre();
  doc_p.parse(stream);
  return operations_p.pos_Operations();
}



boost::shared_ptr<Operations> XmlParser::parse(
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

