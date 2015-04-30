// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern operation_core
#include <boost/test/unit_test.hpp>
#include "fern/language/operation/core/expression_types.h"


BOOST_AUTO_TEST_SUITE(expression_types)

BOOST_AUTO_TEST_CASE(is_satisfied_by)
{
    using namespace fern;
    ExpressionTypes parameter_types, expression_types;

    // constant, unsigned integer parameter
    {
        parameter_types = ExpressionTypes({
            ExpressionType(
                DataTypes::CONSTANT,
                ValueTypes::UNSIGNED_INTEGER)
        });

        expression_types = ExpressionTypes({
            ExpressionType(
                DataTypes::CONSTANT,
                ValueTypes::UNSIGNED_INTEGER)
        });
        BOOST_CHECK(parameter_types.is_satisfied_by(expression_types));
        BOOST_CHECK_EQUAL(parameter_types.id_of_satisfying_type(
            expression_types), 0u);

        expression_types = ExpressionTypes({
            ExpressionType(
                DataTypes::CONSTANT,
                ValueTypes::FLOAT32)
        });
        BOOST_CHECK(!parameter_types.is_satisfied_by(expression_types));
        BOOST_CHECK_EQUAL(parameter_types.id_of_satisfying_type(
            expression_types), 1u);

        expression_types = ExpressionTypes({
            ExpressionType(
                DataTypes::STATIC_FIELD,
                ValueTypes::FLOAT32),
            ExpressionType(
                DataTypes::CONSTANT,
                ValueTypes::UINT8),
            ExpressionType(
                DataTypes::CONSTANT,
                ValueTypes::FLOAT32)
        });
        BOOST_CHECK(parameter_types.is_satisfied_by(expression_types));
        BOOST_CHECK_EQUAL(parameter_types.id_of_satisfying_type(
            expression_types), 1u);
    }
    // constant, uint16
    {
        parameter_types = ExpressionTypes({
            ExpressionType(
                DataTypes::CONSTANT,
                ValueTypes::UINT16)
        });

        expression_types = ExpressionTypes({
            ExpressionType(
                DataTypes::CONSTANT,
                ValueTypes::UNSIGNED_INTEGER)
        });
        // Too broad. Expression properties are not a subset of the parameter's
        // properties. Casting (automatic or explicit) would help here.
        BOOST_CHECK(!parameter_types.is_satisfied_by(expression_types));
        BOOST_CHECK_EQUAL(parameter_types.id_of_satisfying_type(
            expression_types), 1u);
    }

    // constant, unsigned integer parameter
    {
        parameter_types = ExpressionTypes({
            ExpressionType(
                DataTypes::STATIC_FIELD,
                ValueTypes::FLOATING_POINT),
            ExpressionType(
                DataTypes::CONSTANT,
                ValueTypes::UNSIGNED_INTEGER)
        });

        expression_types = ExpressionTypes({
            ExpressionType(
                DataTypes::CONSTANT,
                ValueTypes::UNSIGNED_INTEGER)
        });
        BOOST_CHECK(parameter_types.is_satisfied_by(expression_types));
        BOOST_CHECK_EQUAL(parameter_types.id_of_satisfying_type(
            expression_types), 0u);
    }
}

BOOST_AUTO_TEST_SUITE_END()
