#include "fern/operation/core/expression_type_calculation.h"
#include "fern/operation/core/operation.h"


namespace fern {

static DataTypes result_data_type(
    Operation const& operation,
    size_t index,
    std::vector<DataTypes> const& argument_data_types)
{
    // Data type:
    // Each operation must have a strategy for calculating the data type of
    // the result. It is either a simple combination of the data types of
    // the arguments (C++ rules), or a fixed data type which is an operation
    // property.

    assert(index < operation.results().size());
    assert(argument_data_types.size() == operation.parameters().size());

    {
        Result const& result(operation.results()[index]);
        if(result.expression_type().data_type().fixed()) {
            return result.expression_type().data_type();
        }
    }


    std::vector<DataTypes> data_types(argument_data_types);

    // Select subset of data types that are supported by the parameter.
    for(size_t i = 0; i < data_types.size(); ++i) {
        assert(operation.parameters()[i].expression_types().size() == 1u);
        data_types[i] &=
            operation.parameters()[i].expression_types()[0].data_type();
    }

    // Aggregate all data types.
    DataTypes merged_parameter_data_types;
    for(auto parameter: operation.parameters()) {
        assert(parameter.expression_types().size() == 1u);
        merged_parameter_data_types |=
            parameter.expression_types()[0].data_type();
    }

    DataTypes merged_argument_data_types;
    for(auto data_type: data_types) {
        merged_argument_data_types |= data_type;
    }

    DataTypes result_data_type;

    // Calculate data type of result.
    if(merged_argument_data_types.test(fern::DT_STATIC_FIELD)) {
        result_data_type = DataTypes::STATIC_FIELD;
    }
    else if(merged_argument_data_types.test(fern::DT_CONSTANT)) {
        result_data_type = DataTypes::CONSTANT;
    }
    else if(merged_argument_data_types == DataTypes::UNKNOWN) {
        result_data_type = merged_parameter_data_types;
    }
    else {
        std::cout << "Unhandled data type: " << merged_argument_data_types
            << std::endl;
        assert(false);
    }

    assert(result_data_type.is_subset_of(
        operation.results()[index].expression_type().data_type()));
    return result_data_type;
}


static ValueTypes result_value_type(
    Operation const& operation,
    size_t index,
    std::vector<ValueTypes> const& argument_value_types)
{
    // Value type:
    // Each operation must have a strategy for calculating the value type of
    // the result. It is either a simple combination of the value types of
    // the arguments (C++ rules), or a fixed value type which is an operation
    // property.
    assert(index < operation.results().size());
    assert(argument_value_types.size() == operation.parameters().size());

    {
        Result const& result(operation.results()[index]);
        if(result.expression_type().value_type().fixed()) {
            // The result value type doesn't depend on the argument value
            // types.
            return result.expression_type().value_type();
        }
    }


    if(argument_value_types.size() == 1u && argument_value_types[0].any()) {
        // Propagate the value type(s) of the argument to the result. Assume
        // that the value type of the argument is not changed by the operation.
        return argument_value_types[0];
    }


    // Select the subset of argument value types that is supported by the
    // parameter.
    std::vector<ValueTypes> supported_value_types(argument_value_types);
    for(size_t i = 0; i < supported_value_types.size(); ++i) {
        assert(operation.parameters()[i].expression_types().size() == 1u);
        supported_value_types[i] =
            supported_value_types[i] &=
                operation.parameters()[i].expression_types()[0].value_type();
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
    for(auto parameter: operation.parameters()) {
        assert(parameter.expression_types().size() == 1u);
        merged_parameter_value_types |=
            parameter.expression_types()[0].value_type();
    }


    ValueTypes result_value_type;

    // Calculate value type of result.
    if(merged_argument_value_types.test(fern::VT_FLOAT64)) {
        result_value_type = ValueTypes::FLOAT64;
    }
    else if(merged_argument_value_types.test(fern::VT_FLOAT32)) {
        result_value_type = ValueTypes::FLOAT32;
    }
    else if(merged_argument_value_types.test(fern::VT_UINT64)) {
        result_value_type = ValueTypes::UINT64;
    }
    else if(merged_argument_value_types.test(fern::VT_INT64)) {
        result_value_type = ValueTypes::INT64;
    }
    else if(merged_argument_value_types.test(fern::VT_UINT32)) {
        result_value_type = ValueTypes::UINT32;
    }
    else if(merged_argument_value_types.test(fern::VT_INT32)) {
        result_value_type = ValueTypes::INT32;
    }
    else if(merged_argument_value_types.test(fern::VT_UINT16)) {
        result_value_type = ValueTypes::UINT16;
    }
    else if(merged_argument_value_types.test(fern::VT_UINT8)) {
        result_value_type = ValueTypes::UINT8;
    }
    else if(merged_argument_value_types.test(fern::VT_INT8)) {
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

    assert(result_value_type.is_subset_of(
        operation.results()[index].expression_type().value_type()));
    return result_value_type;
}


//!
/*!
  \tparam    .
  \param     .
  \return    .
  \exception .
  \warning   .
  \sa        .
*/
ExpressionType standard_expression_type(
    Operation const& operation,
    size_t index,
    std::vector<ExpressionType> const& argument_types)
{
    std::vector<DataTypes> data_types;
    std::vector<ValueTypes> value_types;
    for(auto argument_type: argument_types) {
        data_types.push_back(argument_type.data_type());
        value_types.push_back(argument_type.value_type());
    }

    return ExpressionType(
        result_data_type(operation, index, data_types),
        result_value_type(operation, index, value_types));
}

} // namespace fern
