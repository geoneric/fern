#include "fern/operation/xml/operation_xml_parser.h"
#include <memory>
#include <sstream>
#include <stack>
#include <vector>
#include "fern/core/string.h"
#include "fern/operation/core/operations.h"
#include "fern/operation/xml/operation-pskel.hxx"


namespace {

class Operations_pimpl:
    public fern::Operations_pskel
{

private:

    typedef std::vector<std::shared_ptr<fern::Operation>> OperationsData;

    OperationsData   _operations;

public:

    void pre()
    {
        assert(_operations.empty());
    }

    void Operation(
        fern::OperationPtr const& operation)
    {
        _operations.push_back(operation);
    }

    fern::OperationsPtr post_Operations()
    {
        return fern::OperationsPtr(std::make_shared<fern::Operations>(
            _operations));
    }

};


class Operation_pimpl:
    public fern::Operation_pskel
{

private:

    struct OperationData
    {
        fern::String name;
        fern::String description;
        std::vector<fern::Parameter> parameters;
        std::vector<fern::Result> results;
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
        _data_stack.top().name = fern::String(name);
    }

    void Description(
        std::string const& description)
    {
        assert(!_data_stack.empty());
        _data_stack.top().description = fern::String(description);
    }

    void Parameters(
        std::vector<fern::Parameter> const& parameters)
    {
        assert(!_data_stack.empty());
        _data_stack.top().parameters = parameters;
    }

    void Results(
        std::vector<fern::Result> const& results)
    {
        assert(!_data_stack.empty());
        _data_stack.top().results = results;
    }

    fern::OperationPtr post_Operation()
    {
        assert(_data_stack.size() == 1);
        assert(!_data_stack.empty());
        OperationData result(_data_stack.top());
        _data_stack.pop();
        assert(false);
        return fern::OperationPtr();
        // return fern::OperationPtr(new fern::Operation(result.name,
        //     result.description, result.parameters, result.results));
    }

};


class Parameters_pimpl:
    public fern::Parameters_pskel
{

private:

    std::vector<fern::Parameter> _parameters;

public:

    void pre()
    {
        _parameters.clear();
    }

    void Parameter(
        fern::Parameter const& parameter)
    {
        _parameters.push_back(parameter);
    }

    std::vector<fern::Parameter> const& post_Parameters()
    {
        return _parameters;
    }

};


class Parameter_pimpl:
    public fern::Parameter_pskel
{

private:

    struct ParameterData
    {
        fern::String name;
        fern::String description;
        fern::DataTypes data_types;
        fern::ValueTypes value_types;
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
        _data_stack.top().name = fern::String(name);
    }

    void Description(
        std::string const& description)
    {
        assert(!_data_stack.empty());
        _data_stack.top().description = fern::String(description);
    }

    void DataTypes(
        fern::DataTypes const& data_types)
    {
        assert(!_data_stack.empty());
        _data_stack.top().data_types = data_types;
    }

    void ValueTypes(
        fern::ValueTypes const& value_types)
    {
        assert(!_data_stack.empty());
        _data_stack.top().value_types = value_types;
    }

    fern::Parameter post_Parameter()
    {
        assert(_data_stack.size() == 1);
        assert(!_data_stack.empty());
        ParameterData result(_data_stack.top());
        _data_stack.pop();
        return fern::Parameter(result.name, result.description,
            result.data_types, result.value_types);
    }

};


class Results_pimpl:
    public fern::Results_pskel
{

private:

    std::vector<fern::Result> _results;

public:

    void pre()
    {
        _results.clear();
    }

    void Result(
        fern::Result const& result)
    {
        _results.push_back(result);
    }

    std::vector<fern::Result> const& post_Results()
    {
        return _results;
    }

};


class Result_pimpl:
    public fern::Result_pskel
{

private:

    struct ResultData
    {
        fern::String name;
        fern::String description;
        fern::DataTypes data_type;
        fern::ValueTypes value_type;
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
        _data_stack.top().name = fern::String(name);
    }

    void Description(
        std::string const& description)
    {
        assert(!_data_stack.empty());
        _data_stack.top().description = fern::String(description);
    }

    void DataType(
        // fern::DataType const& data_type)
        std::string const& data_type)
    {
        assert(!_data_stack.empty());
        _data_stack.top().data_type = fern::DataTypes::from_string(
            data_type);
    }

    void ValueType(
        // fern::ValueType const& value_type)
        std::string const& value_type)
    {
        assert(!_data_stack.empty());
        _data_stack.top().value_type = fern::ValueTypes::from_string(
            value_type);
    }

    fern::Result post_Result()
    {
        assert(_data_stack.size() == 1);
        assert(!_data_stack.empty());
        ResultData result(_data_stack.top());
        _data_stack.pop();
        return fern::Result(result.name, result.description,
            fern::ExpressionType(result.data_type, result.value_type));
    }

};


class DataTypes_pimpl:
    public fern::DataTypes_pskel
{

private:

    fern::DataTypes _data_types;

public:

    void pre()
    {
        _data_types = fern::DataTypes::UNKNOWN;
    }

    // void DataType(
    //     fern::DataType const& data_type)
    // {
    //     _data_types.push_back(data_type);
    // }

    void DataType(
        std::string const& data_type)
    {
        _data_types |= fern::DataTypes::from_string(data_type);
    }

    fern::DataTypes post_DataTypes()
    {
        return _data_types;
    }

};


// class DataType_pimpl:
//     public fern::DataType_pskel
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
//     fern::DataType post_DataType()
//     {
//         assert(false);
//         assert(_data_type.empty());
//         return DataType::from_string(_data_type);
//     }
// 
// };


class ValueTypes_pimpl:
    public fern::ValueTypes_pskel
{

private:

    fern::ValueTypes _value_types;

public:

    void pre()
    {
        _value_types = fern::ValueTypes::UNKNOWN;
    }

    // void ValueType(
    //     fern::ValueType const& value_type)
    // {
    //     _value_types.push_back(value_type);
    // }

    void ValueType(
        std::string const& value_type)
    {
        _value_types |= fern::ValueTypes::from_string(value_type);
    }

    fern::ValueTypes post_ValueTypes()
    {
        // if(_value_types == fern::ValueTypes::UNKNOWN) {
        //     // No ValueType elements are parsed. Aparently, value type is not
        //     // relevant. This happens for operations dealing with the domain
        //     // only, for example.
        //     _value_types = fern::ValueTypes::NOT_RELEVANT;
        // }
        return _value_types;
    }

};


// class ValueType_pimpl:
//     public fern::ValueType_pskel
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
//     fern::ValueType post_ValueType()
//     {
//         assert(_data_type.empty());
//         return ValueTypes::from_string(_data_type);
//     }
// 
// };

} // Anonymous namespace


namespace fern {

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

} // namespace fern
