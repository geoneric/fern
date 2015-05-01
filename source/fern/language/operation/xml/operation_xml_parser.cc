// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/operation/xml/operation_xml_parser.h"
#include <memory>
#include <sstream>
#include <stack>
#include <string>
#include <vector>
#include "fern/language/operation/core/operations.h"
#include "fern/language/operation/xml/operation-pskel.hxx"


namespace fl = fern::language;


namespace fern {
namespace language {

class Operations_pimpl:
    public Operations_pskel
{

private:

    using OperationsData = std::vector<std::shared_ptr<fl::Operation>>;

    OperationsData   _operations;

public:

    void pre()
    {
        assert(_operations.empty());
    }

    void Operation(
        OperationPtr const& operation)
    {
        _operations.emplace_back(operation);
    }

    OperationsPtr post_Operations()
    {
        return OperationsPtr(std::make_shared<Operations>(
            _operations));
    }

};


class Operation_pimpl:
    public Operation_pskel
{

private:

    struct OperationData
    {
        std::string name;
        std::string description;
        std::vector<Parameter> parameters;
        std::vector<Result> results;
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
        _data_stack.top().name = name;
    }

    void Description(
        std::string const& description)
    {
        assert(!_data_stack.empty());
        _data_stack.top().description = description;
    }

    void Parameters(
        std::vector<Parameter> const& parameters)
    {
        assert(!_data_stack.empty());
        _data_stack.top().parameters = parameters;
    }

    void Results(
        std::vector<Result> const& results)
    {
        assert(!_data_stack.empty());
        _data_stack.top().results = results;
    }

    OperationPtr post_Operation()
    {
        assert(_data_stack.size() == 1);
        assert(!_data_stack.empty());
        OperationData result(_data_stack.top());
        _data_stack.pop();
        assert(false);
        return OperationPtr();
        // return OperationPtr(new Operation(result.name,
        //     result.description, result.parameters, result.results));
    }

};


class Parameters_pimpl:
    public Parameters_pskel
{

private:

    std::vector<fl::Parameter> _parameters;

public:

    void pre()
    {
        _parameters.clear();
    }

    void Parameter(
        fl::Parameter const& parameter)
    {
        _parameters.emplace_back(parameter);
    }

    std::vector<fl::Parameter> const& post_Parameters()
    {
        return _parameters;
    }

};


class Parameter_pimpl:
    public Parameter_pskel
{

private:

    struct ParameterData
    {
        std::string name;
        std::string description;
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
        _data_stack.top().name = name;
    }

    void Description(
        std::string const& description)
    {
        assert(!_data_stack.empty());
        _data_stack.top().description = description;
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

    Parameter post_Parameter()
    {
        assert(_data_stack.size() == 1);
        assert(!_data_stack.empty());
        ParameterData result(_data_stack.top());
        _data_stack.pop();
        return Parameter(result.name, result.description,
            result.data_types, result.value_types);
    }

};


class Results_pimpl:
    public Results_pskel
{

private:

    std::vector<fl::Result> _results;

public:

    void pre()
    {
        _results.clear();
    }

    void Result(
        fl::Result const& result)
    {
        _results.emplace_back(result);
    }

    std::vector<fl::Result> const& post_Results()
    {
        return _results;
    }

};


class Result_pimpl:
    public Result_pskel
{

private:

    struct ResultData
    {
        std::string name;
        std::string description;
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
        _data_stack.top().name = name;
    }

    void Description(
        std::string const& description)
    {
        assert(!_data_stack.empty());
        _data_stack.top().description = description;
    }

    void DataType(
        // DataType const& data_type)
        std::string const& data_type)
    {
        assert(!_data_stack.empty());
        _data_stack.top().data_type = DataTypes::from_string(
            data_type);
    }

    void ValueType(
        // ValueType const& value_type)
        std::string const& value_type)
    {
        assert(!_data_stack.empty());
        _data_stack.top().value_type = ValueTypes::from_string(
            value_type);
    }

    Result post_Result()
    {
        assert(_data_stack.size() == 1);
        assert(!_data_stack.empty());
        ResultData result(_data_stack.top());
        _data_stack.pop();
        return Result(result.name, result.description,
            ExpressionType(result.data_type, result.value_type));
    }

};


class DataTypes_pimpl:
    public DataTypes_pskel
{

private:

    DataTypes _data_types;

public:

    void pre()
    {
        _data_types = DataTypes::UNKNOWN;
    }

    // void DataType(
    //     DataType const& data_type)
    // {
    //     _data_types.emplace_back(data_type);
    // }

    void DataType(
        std::string const& data_type)
    {
        _data_types |= DataTypes::from_string(data_type);
    }

    DataTypes post_DataTypes()
    {
        return _data_types;
    }

};


// class DataType_pimpl:
//     public DataType_pskel
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
//     DataType post_DataType()
//     {
//         assert(false);
//         assert(_data_type.empty());
//         return DataType::from_string(_data_type);
//     }
// 
// };


class ValueTypes_pimpl:
    public ValueTypes_pskel
{

private:

    ValueTypes _value_types;

public:

    void pre()
    {
        _value_types = ValueTypes::UNKNOWN;
    }

    // void ValueType(
    //     ValueType const& value_type)
    // {
    //     _value_types.emplace_back(value_type);
    // }

    void ValueType(
        std::string const& value_type)
    {
        _value_types |= ValueTypes::from_string(value_type);
    }

    ValueTypes post_ValueTypes()
    {
        // if(_value_types == ValueTypes::UNKNOWN) {
        //     // No ValueType elements are parsed. Aparently, value type is not
        //     // relevant. This happens for operations dealing with the domain
        //     // only, for example.
        //     _value_types = ValueTypes::NOT_RELEVANT;
        // }
        return _value_types;
    }

};


// class ValueType_pimpl:
//     public ValueType_pskel
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
//     ValueType post_ValueType()
//     {
//         assert(_data_type.empty());
//         return ValueTypes::from_string(_data_type);
//     }
// 
// };


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
    std::string const& xml) const
{
    // Copy string contents in a string stream and work with that.
    std::stringstream stream;
    stream.exceptions(std::ifstream::badbit | std::ifstream::failbit);
    stream << xml; // << std::endl;

    return parse(stream);
}

} // namespace language
} // namespace fern
