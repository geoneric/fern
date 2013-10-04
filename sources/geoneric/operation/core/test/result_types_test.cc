#define BOOST_TEST_MODULE geoneric operation_core
#include <boost/test/unit_test.hpp>
#include "geoneric/operation/core/result_types.h"


BOOST_AUTO_TEST_SUITE(result_types)

BOOST_AUTO_TEST_CASE(is_satisfied_by)
{
    using namespace geoneric;
    ResultTypes parameter_types, expression_types;

    // constant, unsigned integer parameter
    {
        parameter_types = ResultTypes({
            ResultType(
                DataTypes::CONSTANT,
                ValueTypes::UNSIGNED_INTEGER)
        });

        expression_types = ResultTypes({
            ResultType(
                DataTypes::CONSTANT,
                ValueTypes::UNSIGNED_INTEGER)
        });
        BOOST_CHECK(parameter_types.is_satisfied_by(expression_types));
        BOOST_CHECK_EQUAL(parameter_types.id_of_satisfying_type(
            expression_types), 0u);

        expression_types = ResultTypes({
            ResultType(
                DataTypes::CONSTANT,
                ValueTypes::FLOAT32)
        });
        BOOST_CHECK(!parameter_types.is_satisfied_by(expression_types));
        BOOST_CHECK_EQUAL(parameter_types.id_of_satisfying_type(
            expression_types), 1u);

        expression_types = ResultTypes({
            ResultType(
                DataTypes::STATIC_FIELD,
                ValueTypes::FLOAT32),
            ResultType(
                DataTypes::CONSTANT,
                ValueTypes::UINT8),
            ResultType(
                DataTypes::CONSTANT,
                ValueTypes::FLOAT32)
        });
        BOOST_CHECK(parameter_types.is_satisfied_by(expression_types));
        BOOST_CHECK_EQUAL(parameter_types.id_of_satisfying_type(
            expression_types), 1u);
    }
    // constant, uint16
    {
        parameter_types = ResultTypes({
            ResultType(
                DataTypes::CONSTANT,
                ValueTypes::UINT16)
        });

        expression_types = ResultTypes({
            ResultType(
                DataTypes::CONSTANT,
                ValueTypes::UNSIGNED_INTEGER)
        });
        // Too broad. Expression properties are not a subset of the parameter's
        // properties. Casting (automatically or explicit) would help here.
        BOOST_CHECK(!parameter_types.is_satisfied_by(expression_types));
        BOOST_CHECK_EQUAL(parameter_types.id_of_satisfying_type(
            expression_types), 1u);
    }

    // constant, unsigned integer parameter
    {
        parameter_types = ResultTypes({
            ResultType(
                DataTypes::STATIC_FIELD,
                ValueTypes::FLOATING_POINT),
            ResultType(
                DataTypes::CONSTANT,
                ValueTypes::UNSIGNED_INTEGER)
        });

        expression_types = ResultTypes({
            ResultType(
                DataTypes::CONSTANT,
                ValueTypes::UNSIGNED_INTEGER)
        });
        BOOST_CHECK(parameter_types.is_satisfied_by(expression_types));
        BOOST_CHECK_EQUAL(parameter_types.id_of_satisfying_type(
            expression_types), 0u);
    }
}

BOOST_AUTO_TEST_SUITE_END()
