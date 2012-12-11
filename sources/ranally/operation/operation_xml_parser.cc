#include "ranally/operation/operation_xml_parser.h"
#include <memory>
#include <sstream>
#include <stack>
#include <vector>
#include "ranally/core/string.h"
#include "ranally/operation/operation-pskel.hxx"
#include "ranally/operation/operations.h"


namespace {

class Operations_pimpl:
    public ranally::Operations_pskel
{

private:

    typedef std::vector<std::shared_ptr<ranally::Operation>> OperationsData;

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
        return ranally::OperationsPtr(new ranally::Operations(_operations));
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

    std::stack<OperationData> _data_stack;

public:

    void pre()
    {
        assert(_data_stack.empty());
        _data_stack.push(OperationData());
    }

    void Name(
        std::string const& name)
    {
        assert(!_data_stack.empty());
        _data_stack.top().name = ranally::String(name);
    }

    void Description(
        std::string const& description)
    {
        assert(!_data_stack.empty());
        _data_stack.top().description = ranally::String(description);
    }

    void Parameters(
        std::vector<ranally::Parameter> const& parameters)
    {
        assert(!_data_stack.empty());
        _data_stack.top().parameters = parameters;
    }

    void Results(
        std::vector<ranally::Result> const& results)
    {
        assert(!_data_stack.empty());
        _data_stack.top().results = results;
    }

    ranally::OperationPtr post_Operation()
    {
        assert(_data_stack.size() == 1);
        assert(!_data_stack.empty());
        OperationData result(_data_stack.top());
        _data_stack.pop();
        return ranally::OperationPtr(new ranally::Operation(result.name,
            result.description, result.parameters, result.results));
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
        ranally::DataTypes data_types;
        ranally::ValueTypes value_types;
    };

    std::stack<ParameterData> _data_stack;

public:

    void pre()
    {
        assert(_data_stack.empty());
        _data_stack.push(ParameterData());
    }

    void Name(
        std::string const& name)
    {
        assert(!_data_stack.empty());
        _data_stack.top().name = ranally::String(name);
    }

    void Description(
        std::string const& description)
    {
        assert(!_data_stack.empty());
        _data_stack.top().description = ranally::String(description);
    }

    void DataTypes(
        ranally::DataTypes const& data_types)
    {
        assert(!_data_stack.empty());
        _data_stack.top().data_types = data_types;
    }

    void ValueTypes(
        ranally::ValueTypes const& value_types)
    {
        assert(!_data_stack.empty());
        _data_stack.top().value_types = value_types;
    }

    ranally::Parameter post_Parameter()
    {
        assert(_data_stack.size() == 1);
        assert(!_data_stack.empty());
        ParameterData result(_data_stack.top());
        _data_stack.pop();
        return ranally::Parameter(result.name, result.description,
            result.data_types, result.value_types);
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
        ranally::DataTypes data_type;
        ranally::ValueTypes value_type;
    };

    std::stack<ResultData> _data_stack;

public:

    void pre()
    {
        assert(_data_stack.empty());
        _data_stack.push(ResultData());
    }

    void Name(
        std::string const& name)
    {
        assert(!_data_stack.empty());
        _data_stack.top().name = ranally::String(name);
    }

    void Description(
        std::string const& description)
    {
        assert(!_data_stack.empty());
        _data_stack.top().description = ranally::String(description);
    }

    void DataType(
        // ranally::DataType const& data_type)
        std::string const& data_type)
    {
        assert(!_data_stack.empty());
        _data_stack.top().data_type = ranally::DataTypes::from_string(
            data_type);
    }

    void ValueType(
        // ranally::ValueType const& value_type)
        std::string const& value_type)
    {
        assert(!_data_stack.empty());
        _data_stack.top().value_type = ranally::ValueTypes::from_string(
            value_type);
    }

    ranally::Result post_Result()
    {
        assert(_data_stack.size() == 1);
        assert(!_data_stack.empty());
        ResultData result(_data_stack.top());
        _data_stack.pop();
        return ranally::Result(result.name, result.description,
          result.data_type, result.value_type);
    }

};


class DataTypes_pimpl:
    public ranally::DataTypes_pskel
{

private:

    ranally::DataTypes _data_types;

public:

    void pre()
    {
        _data_types = ranally::DataTypes::UNKNOWN;
    }

    // void DataType(
    //     ranally::DataType const& data_type)
    // {
    //     _data_types.push_back(data_type);
    // }

    void DataType(
        std::string const& data_type)
    {
        _data_types |= ranally::DataTypes::from_string(data_type);
    }

    ranally::DataTypes post_DataTypes()
    {
        return _data_types;
    }

};


// class DataType_pimpl:
//     public ranally::DataType_pskel
// {
// 
// private:
// 
//     std::string      _data_type;
// 
// public:
// 
//     void DataType(
//         std::string const& data_type)
//     {
//         assert(false);
//         _data_type = data_type;
//     }
// 
//     std::string post_string()
//     {
//         assert(false);
//         return _data_type;
//     }
// 
//     ranally::DataType post_DataType()
//     {
//         assert(false);
//         assert(_data_type.empty());
//         return DataType::from_string(_data_type);
//     }
// 
// };


class ValueTypes_pimpl:
    public ranally::ValueTypes_pskel
{

private:

    ranally::ValueTypes _value_types;

public:

    void pre()
    {
        _value_types = ranally::ValueTypes::UNKNOWN;
    }

    // void ValueType(
    //     ranally::ValueType const& value_type)
    // {
    //     _value_types.push_back(value_type);
    // }

    void ValueType(
        std::string const& value_type)
    {
        _value_types |= ranally::ValueTypes::from_string(value_type);
    }

    ranally::ValueTypes post_ValueTypes()
    {
        // if(_value_types == ranally::ValueTypes::UNKNOWN) {
        //     // No ValueType elements are parsed. Aparently, value type is not
        //     // relevant. This happens for operations dealing with the domain
        //     // only, for example.
        //     _value_types = ranally::ValueTypes::NOT_RELEVANT;
        // }
        return _value_types;
    }

};


// class ValueType_pimpl:
//     public ranally::ValueType_pskel
// {
// 
// private:
// 
//     std::string      _data_type;
// 
// public:
// 
//     void ValueType(
//         std::string const& data_type)
//     {
//         _data_type = data_type;
//     }
// 
//     std::string post_string()
//     {
//         return _data_type;
//     }
// 
//     ranally::ValueType post_ValueType()
//     {
//         assert(_data_type.empty());
//         return ValueTypes::from_string(_data_type);
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

    // DataType_pimpl data_type_p;
    // ValueType_pimpl value_type_p;

    DataTypes_pimpl data_types_p;
    data_types_p.parsers(string_p /* data_type_p */);

    ValueTypes_pimpl value_types_p;
    value_types_p.parsers(string_p /* value_type_p */);

    Parameter_pimpl parameter_p;
    parameter_p.parsers(string_p, string_p, data_types_p, value_types_p);

    Parameters_pimpl parameters_p;
    parameters_p.parsers(parameter_p);

    Result_pimpl result_p;
    result_p.parsers(string_p, string_p, string_p, string_p
        /* data_type_p, value_type_p */);

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
    stream << xml.encode_in_utf8(); // << std::endl;

    return parse(stream);
}

} // namespace ranally
