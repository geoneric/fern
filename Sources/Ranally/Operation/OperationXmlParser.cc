#include "Ranally/Operation/OperationXmlParser.h"
#include <sstream>
#include <stack>
#include <vector>
#include <boost/make_shared.hpp>
#include "Ranally/Util/String.h"
#include "Operation-pskel.hxx"
#include "Ranally/Operation/DataType.h"
#include "Ranally/Operation/Operations.h"


namespace {

ranally::DataType stringToDataType(
    std::string const& string)
{
    assert(!string.empty());
    ranally::DataType dataType = ranally::DT_UNKNOWN;

    if(string == "Value") {
        dataType = ranally::DT_VALUE;
    }
    else if(string == "Raster") {
        dataType = ranally::DT_RASTER;
    }
    else if(string == "Feature") {
        dataType = ranally::DT_FEATURE;
    }
    else if(string == "Spatial") {
        dataType = ranally::DT_SPATIAL;
    }
    else if(string == "All") {
        dataType = ranally::DT_ALL;
    }
    else if(string == "DependsOnInput") {
        dataType = ranally::DT_DEPENDS_ON_INPUT;
    }

    assert(dataType != ranally::DT_UNKNOWN);
    return dataType;
}


static ranally::ValueType stringToValueType(
    std::string const& string)
{
    assert(!string.empty());
    ranally::ValueType valueType = ranally::VT_UNKNOWN;

    if(string == "UInt8") {
        valueType = ranally::VT_UINT8;
    }
    else if(string == "Int8") {
        valueType = ranally::VT_INT8;
    }
    else if(string == "UInt16") {
        valueType = ranally::VT_UINT16;
    }
    else if(string == "Int16") {
        valueType = ranally::VT_INT16;
    }
    else if(string == "UInt32") {
        valueType = ranally::VT_UINT32;
    }
    else if(string == "Int32") {
        valueType = ranally::VT_INT32;
    }
    else if(string == "UInt64") {
        valueType = ranally::VT_UINT64;
    }
    else if(string == "Int64") {
        valueType = ranally::VT_INT64;
    }
    else if(string == "Float32") {
        valueType = ranally::VT_FLOAT32;
    }
    else if(string == "Float64") {
        valueType = ranally::VT_FLOAT64;
    }
    else if(string == "String") {
        valueType = ranally::VT_STRING;
    }
    else if(string == "UnsignedInteger") {
        valueType = ranally::VT_UNSIGNED_INTEGER;
    }
    else if(string == "SignedInteger") {
        valueType = ranally::VT_SIGNED_INTEGER;
    }
    else if(string == "Integer") {
        valueType = ranally::VT_INTEGER;
    }
    else if(string == "FloatingPoint") {
        valueType = ranally::VT_FLOATING_POINT;
    }
    else if(string == "Number") {
        valueType = ranally::VT_NUMBER;
    }
    else if(string == "All") {
        valueType = ranally::VT_ALL;
    }
    else if(string == "DependsOnInput") {
        valueType = ranally::VT_DEPENDS_ON_INPUT;
    }

    assert(valueType != ranally::VT_UNKNOWN);
    return valueType;
}


class Operations_pimpl:
    public ranally::Operations_pskel
{

private:

    typedef std::vector<boost::shared_ptr<ranally::Operation> >
        OperationsData;

    OperationsData   _operations;

public:

    void pre()
    {
        assert(_operations.empty());
    }

    void Operation(
        ranally::OperationPtr const& operation)
    {
        _operations.push_back(operation);
    }

    ranally::OperationsPtr post_Operations()
    {
        return boost::make_shared<ranally::Operations>(_operations);
    }

};


class Operation_pimpl:
    public ranally::Operation_pskel
{

private:

    struct OperationData
    {
        ranally::String name;
        ranally::String description;
        std::vector<ranally::Parameter> parameters;
        std::vector<ranally::Result> results;
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
        _dataStack.top().name = ranally::String(name);
    }

    void Description(
        std::string const& description)
    {
        assert(!_dataStack.empty());
        _dataStack.top().description = ranally::String(description);
    }

    void Parameters(
        std::vector<ranally::Parameter> const& parameters)
    {
        assert(!_dataStack.empty());
        _dataStack.top().parameters = parameters;
    }

    void Results(
        std::vector<ranally::Result> const& results)
    {
        assert(!_dataStack.empty());
        _dataStack.top().results = results;
    }

    ranally::OperationPtr post_Operation()
    {
        assert(_dataStack.size() == 1);
        assert(!_dataStack.empty());
        OperationData result(_dataStack.top());
        _dataStack.pop();
        return boost::make_shared<ranally::Operation>(result.name,
            result.description, result.parameters, result.results);
    }

};


class Parameters_pimpl:
    public ranally::Parameters_pskel
{

private:

    std::vector<ranally::Parameter> _parameters;

public:

    void pre()
    {
        _parameters.clear();
    }

    void Parameter(
        ranally::Parameter const& parameter)
    {
        _parameters.push_back(parameter);
    }

    std::vector<ranally::Parameter> const& post_Parameters()
    {
        return _parameters;
    }

};


class Parameter_pimpl:
    public ranally::Parameter_pskel
{

private:

    struct ParameterData
    {
        ranally::String name;
        ranally::String description;
        ranally::DataTypes dataTypes;
        ranally::ValueTypes valueTypes;
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
        _dataStack.top().name = ranally::String(name);
    }

    void Description(
        std::string const& description)
    {
        assert(!_dataStack.empty());
        _dataStack.top().description = ranally::String(description);
    }

    void DataTypes(
        ranally::DataTypes const& dataTypes)
    {
        assert(!_dataStack.empty());
        _dataStack.top().dataTypes = dataTypes;
    }

    void ValueTypes(
        ranally::ValueTypes const& valueTypes)
    {
        assert(!_dataStack.empty());
        _dataStack.top().valueTypes = valueTypes;
    }

    ranally::Parameter post_Parameter()
    {
        assert(_dataStack.size() == 1);
        assert(!_dataStack.empty());
        ParameterData result(_dataStack.top());
        _dataStack.pop();
        return ranally::Parameter(result.name, result.description,
            result.dataTypes, result.valueTypes);
    }

};


class Results_pimpl:
    public ranally::Results_pskel
{

private:

    std::vector<ranally::Result> _results;

public:

    void pre()
    {
        _results.clear();
    }

    void Result(
        ranally::Result const& result)
    {
        _results.push_back(result);
    }

    std::vector<ranally::Result> const& post_Results()
    {
        return _results;
    }

};


class Result_pimpl:
    public ranally::Result_pskel
{

private:

    struct ResultData
    {
        ranally::String name;
        ranally::String description;
        ranally::DataType dataType;
        ranally::ValueType valueType;
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
        _dataStack.top().name = ranally::String(name);
    }

    void Description(
        std::string const& description)
    {
        assert(!_dataStack.empty());
        _dataStack.top().description = ranally::String(description);
    }

    void DataType(
        // ranally::DataType const& dataType)
        std::string const& dataType)
    {
        assert(!_dataStack.empty());
        _dataStack.top().dataType = stringToDataType(dataType);
    }

    void ValueType(
        // ranally::ValueType const& valueType)
        std::string const& valueType)
    {
        assert(!_dataStack.empty());
        _dataStack.top().valueType = stringToValueType(valueType);
    }

    ranally::Result post_Result()
    {
        assert(_dataStack.size() == 1);
        assert(!_dataStack.empty());
        ResultData result(_dataStack.top());
        _dataStack.pop();
        return ranally::Result(result.name, result.description,
          result.dataType, result.valueType);
    }

};


class DataTypes_pimpl:
    public ranally::DataTypes_pskel
{

private:

    ranally::DataTypes _dataTypes;

public:

    void pre()
    {
        _dataTypes = ranally::DT_UNKNOWN;
    }

    // void DataType(
    //     ranally::DataType const& dataType)
    // {
    //     _dataTypes.push_back(dataType);
    // }

    void DataType(
        std::string const& dataType)
    {
        _dataTypes |= stringToDataType(dataType);
    }

    ranally::DataTypes post_DataTypes()
    {
        return _dataTypes;
    }

};


// class DataType_pimpl:
//     public ranally::DataType_pskel
// {
// 
// private:
// 
//     std::string      _dataType;
// 
// public:
// 
//     void DataType(
//         std::string const& dataType)
//     {
//         assert(false);
//         _dataType = dataType;
//     }
// 
//     std::string post_string()
//     {
//         assert(false);
//         return _dataType;
//     }
// 
//     ranally::DataType post_DataType()
//     {
//         assert(false);
//         assert(_dataType.empty());
//         return stringToDataType(_dataType);
//     }
// 
// };


class ValueTypes_pimpl:
    public ranally::ValueTypes_pskel
{

private:

    ranally::ValueTypes _valueTypes;

public:

    void pre()
    {
        _valueTypes = ranally::VT_UNKNOWN;
    }

    // void ValueType(
    //     ranally::ValueType const& valueType)
    // {
    //     _valueTypes.push_back(valueType);
    // }

    void ValueType(
        std::string const& valueType)
    {
        _valueTypes |= stringToValueType(valueType);
    }

    ranally::ValueTypes post_ValueTypes()
    {
        return _valueTypes;
    }

};


// class ValueType_pimpl:
//     public ranally::ValueType_pskel
// {
// 
// private:
// 
//     std::string      _dataType;
// 
// public:
// 
//     void ValueType(
//         std::string const& dataType)
//     {
//         _dataType = dataType;
//     }
// 
//     std::string post_string()
//     {
//         return _dataType;
//     }
// 
//     ranally::ValueType post_ValueType()
//     {
//         assert(_dataType.empty());
//         return stringToValueType(_dataType);
//     }
// 
// };

} // Anonymous namespace


namespace ranally {

OperationXmlParser::OperationXmlParser()
{
}



OperationXmlParser::~OperationXmlParser()
{
}



OperationsPtr OperationXmlParser::parse(
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


OperationsPtr OperationXmlParser::parse(
    String const& xml) const
{
    // Copy string contents in a string stream and work with that.
    std::stringstream stream;
    stream.exceptions(std::ifstream::badbit | std::ifstream::failbit);
    stream << xml.encodeInUTF8(); // << std::endl;

    return parse(stream);
}

} // namespace ranally
