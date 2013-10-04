#define BOOST_TEST_MODULE geoneric operation_core
#include <boost/test/unit_test.hpp>
#include "geoneric/operation/core/result_type.h"


BOOST_AUTO_TEST_SUITE(result_type)

BOOST_AUTO_TEST_CASE(result_type)
{
    geoneric::ResultType result_type(geoneric::DataTypes::CONSTANT,
        geoneric::ValueTypes::INT32);
    BOOST_CHECK_EQUAL(result_type.data_type(), geoneric::DataTypes::CONSTANT);
    BOOST_CHECK_EQUAL(result_type.value_type(), geoneric::ValueTypes::INT32);

    {
        geoneric::ResultType copy(result_type);
        BOOST_CHECK_EQUAL(result_type.data_type(),
            geoneric::DataTypes::CONSTANT);
        BOOST_CHECK_EQUAL(result_type.value_type(),
            geoneric::ValueTypes::INT32);
    }

    {
        geoneric::ResultType copy = result_type;
        BOOST_CHECK_EQUAL(result_type.data_type(),
            geoneric::DataTypes::CONSTANT);
        BOOST_CHECK_EQUAL(result_type.value_type(), geoneric::ValueTypes::INT32);
    }
}

BOOST_AUTO_TEST_CASE(is_satisfied_by)
{
    using namespace geoneric;
    ResultType parameter_type, expression_type;

    // constant, int32 parameter
    {
        parameter_type = ResultType(
            DataTypes::CONSTANT,
            ValueTypes::INT32);

        expression_type = ResultType();
        BOOST_CHECK(!parameter_type.is_satisfied_by(expression_type));

        expression_type = ResultType(
            DataTypes::CONSTANT,
            ValueTypes::INT32);
        BOOST_CHECK(parameter_type.is_satisfied_by(expression_type));

        expression_type = ResultType(
            DataTypes::CONSTANT,
            ValueTypes::INT64);
        BOOST_CHECK(!parameter_type.is_satisfied_by(expression_type));

        expression_type = ResultType(
            DataTypes::STATIC_FIELD,
            ValueTypes::INT32);
        BOOST_CHECK(!parameter_type.is_satisfied_by(expression_type));

        expression_type = ResultType(
            DataTypes::CONSTANT,
            ValueTypes::INTEGER);
        // Too broad. Expression properties are not a subset of the parameter's
        // properties. Casting (automatically or explicit) would help here.
        BOOST_CHECK(!parameter_type.is_satisfied_by(expression_type));
    }
}

BOOST_AUTO_TEST_SUITE_END()
