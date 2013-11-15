#define BOOST_TEST_MODULE geoneric operation_core
#include <boost/test/unit_test.hpp>
#include "fern/core/expression_type.h"


BOOST_AUTO_TEST_SUITE(expression_type)

BOOST_AUTO_TEST_CASE(expression_type)
{
    fern::ExpressionType expression_type(fern::DataTypes::CONSTANT,
        fern::ValueTypes::INT32);
    BOOST_CHECK_EQUAL(expression_type.data_type(),
        fern::DataTypes::CONSTANT);
    BOOST_CHECK_EQUAL(expression_type.value_type(),
        fern::ValueTypes::INT32);

    {
        fern::ExpressionType copy(expression_type);
        BOOST_CHECK_EQUAL(expression_type.data_type(),
            fern::DataTypes::CONSTANT);
        BOOST_CHECK_EQUAL(expression_type.value_type(),
            fern::ValueTypes::INT32);
    }

    {
        fern::ExpressionType copy = expression_type;
        BOOST_CHECK_EQUAL(expression_type.data_type(),
            fern::DataTypes::CONSTANT);
        BOOST_CHECK_EQUAL(expression_type.value_type(), fern::ValueTypes::INT32);
    }
}

BOOST_AUTO_TEST_CASE(is_satisfied_by)
{
    using namespace fern;
    ExpressionType parameter_type, expression_type;

    // constant, int32 parameter
    {
        parameter_type = ExpressionType(
            DataTypes::CONSTANT,
            ValueTypes::INT32);

        expression_type = ExpressionType();
        BOOST_CHECK(!parameter_type.is_satisfied_by(expression_type));

        expression_type = ExpressionType(
            DataTypes::CONSTANT,
            ValueTypes::INT32);
        BOOST_CHECK(parameter_type.is_satisfied_by(expression_type));

        expression_type = ExpressionType(
            DataTypes::CONSTANT,
            ValueTypes::INT64);
        BOOST_CHECK(!parameter_type.is_satisfied_by(expression_type));

        expression_type = ExpressionType(
            DataTypes::STATIC_FIELD,
            ValueTypes::INT32);
        BOOST_CHECK(!parameter_type.is_satisfied_by(expression_type));

        expression_type = ExpressionType(
            DataTypes::CONSTANT,
            ValueTypes::INTEGER);
        // Too broad. Expression properties are not a subset of the parameter's
        // properties. Casting (automatically or explicit) would help here.
        BOOST_CHECK(!parameter_type.is_satisfied_by(expression_type));
    }
}

BOOST_AUTO_TEST_SUITE_END()
