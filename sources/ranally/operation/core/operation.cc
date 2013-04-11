#include "ranally/operation/core/operation.h"
#include "ranally/operation/core/parameter.h"
#include "ranally/operation/core/result.h"


namespace ranally {

Operation::Operation(
    String const& name,
    String const& description,
    std::vector<Parameter> const& parameters,
    std::vector<Result> const& results)

    : _name(name),
      _description(description),
      _parameters(parameters),
      _results(results)

{
    assert(!name.is_empty());
    assert(!description.is_empty());
}


Operation::Operation(
    Operation const& other)

    : _name(other._name),
      _description(other._description),
      _parameters(other._parameters),
      _results(other._results)

{
}


Operation& Operation::operator=(
    Operation const& other)
{
    if(&other != this) {
        _name = other._name;
        _description = other._description;
        _parameters = other._parameters;
        _results = other._results;
    }

    return *this;
}


String const& Operation::name() const
{
    return _name;
}


String const& Operation::description() const
{
    return _description;
}


size_t Operation::arity() const
{
    return _parameters.size();
}


std::vector<Parameter> const& Operation::parameters() const
{
    return _parameters;
}


std::vector<Result> const& Operation::results() const
{
    return _results;
}


DataTypes Operation::result_data_type(
    size_t index,
    std::vector<DataTypes> const& argument_data_types) const
{
    // Data type:
    // Each operation must have a strategy for calculating the data type of
    // the result. It is either a simple combination of the data types of
    // the arguments (C++ rules), or a fixed data type which is an operation
    // property.

    assert(index < _results.size());
    assert(argument_data_types.size() == _parameters.size());

    {
        Result const& result(_results[index]);
        if(result.data_type().fixed()) {
            return result.data_type();
        }
    }


    std::vector<DataTypes> data_types(argument_data_types);

    // Select subset of data types that are supported by the parameter.
    for(size_t i = 0; i < data_types.size(); ++i) {
        data_types[i] = data_types[i] &= _parameters[i].data_types();
    }

    // Aggregate all data types.
    DataTypes merged_parameter_data_types;
    for(auto parameter: _parameters) {
        merged_parameter_data_types |= parameter.data_types();
    }

    DataTypes merged_argument_data_types;
    for(auto data_type: data_types) {
        merged_argument_data_types |= data_type;
    }

    DataTypes result_data_type;

    // Calculate data type of result.
    if(merged_argument_data_types.test(ranally::DT_SCALAR)) {
        result_data_type = DataTypes::SCALAR;
    }
    else if(merged_argument_data_types == DataTypes::UNKNOWN) {
        result_data_type = merged_parameter_data_types;
    }
    else {
        std::cout << "Unhandled data type: " << merged_argument_data_types
            << std::endl;
        assert(false);
    }

    return result_data_type;
}


ValueTypes Operation::result_value_type(
    size_t index,
    std::vector<ValueTypes> const& argument_value_types) const
{
    // Value type:
    // Each operation must have a strategy for calculating the value type of
    // the result. It is either a simple combination of the value types of
    // the arguments (C++ rules), or a fixed value type which is an operation
    // property.
    assert(index < _results.size());
    assert(argument_value_types.size() == _parameters.size());


    {
        Result const& result(_results[index]);
        if(result.value_type().fixed()) {
            return result.value_type();
        }
    }


    // Select subset of value types that are supported by the parameter.
    std::vector<ValueTypes> supported_value_types(argument_value_types);
    for(size_t i = 0; i < supported_value_types.size(); ++i) {
        supported_value_types[i] =
            supported_value_types[i] &= _parameters[i].value_types();
    }


    ValueTypes merged_argument_value_types;
    {
        for(auto value_type: supported_value_types) {
            merged_argument_value_types |= value_type;
        }

        if(merged_argument_value_types.fixed()) {
            return merged_argument_value_types;
        }
    }


    // Aggregate all value types.
    ValueTypes merged_parameter_value_types;
    for(auto parameter: _parameters) {
        merged_parameter_value_types |= parameter.value_types();
    }


    ValueTypes result_value_type;

    // Calculate value type of result.
    if(merged_argument_value_types.test(ranally::VT_FLOAT64)) {
        result_value_type = ValueTypes::FLOAT64;
    }
    else if(merged_argument_value_types.test(ranally::VT_FLOAT32)) {
        result_value_type = ValueTypes::FLOAT32;
    }
    else if(merged_argument_value_types.test(ranally::VT_UINT64)) {
        result_value_type = ValueTypes::UINT64;
    }
    else if(merged_argument_value_types.test(ranally::VT_INT64)) {
        result_value_type = ValueTypes::INT64;
    }
    else if(merged_argument_value_types.test(ranally::VT_UINT32)) {
        result_value_type = ValueTypes::UINT32;
    }
    else if(merged_argument_value_types.test(ranally::VT_INT32)) {
        result_value_type = ValueTypes::INT32;
    }
    else if(merged_argument_value_types.test(ranally::VT_UINT16)) {
        result_value_type = ValueTypes::UINT16;
    }
    else if(merged_argument_value_types.test(ranally::VT_INT8)) {
        result_value_type = ValueTypes::INT8;
    }
    else if(merged_argument_value_types == ValueTypes::UNKNOWN) {
        result_value_type = merged_parameter_value_types;
    }
    else {
        std::cout << "Unhandled value type: " << merged_argument_value_types
            << std::endl;
        assert(false);
    }

    return result_value_type;
}


ResultType Operation::result_type(
    size_t index,
    std::vector<ResultType> const& argument_types) const
{
    std::vector<DataTypes> data_types;
    std::vector<ValueTypes> value_types;
    for(auto argument_type: argument_types) {
        data_types.push_back(argument_type.data_type());
        value_types.push_back(argument_type.value_type());
    }

    return ResultType(
        result_data_type(index, data_types),
        result_value_type(index, value_types));
}

} // namespace ranally
