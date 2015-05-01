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

namespace fl = fern::language;


BOOST_AUTO_TEST_SUITE(expression_types)

BOOST_AUTO_TEST_CASE(is_satisfied_by)
{
    fl::ExpressionTypes parameter_types, expression_types;

    // constant, unsigned integer parameter
    {
        parameter_types = fl::ExpressionTypes({
            fern::ExpressionType(
                fern::DataTypes::CONSTANT,
                fern::ValueTypes::UNSIGNED_INTEGER)
        });

        expression_types = fl::ExpressionTypes({
            fern::ExpressionType(
                fern::DataTypes::CONSTANT,
                fern::ValueTypes::UNSIGNED_INTEGER)
        });
        BOOST_CHECK(parameter_types.is_satisfied_by(expression_types));
        BOOST_CHECK_EQUAL(parameter_types.id_of_satisfying_type(
            expression_types), 0u);

        expression_types = fl::ExpressionTypes({
            fern::ExpressionType(
                fern::DataTypes::CONSTANT,
                fern::ValueTypes::FLOAT32)
        });
        BOOST_CHECK(!parameter_types.is_satisfied_by(expression_types));
        BOOST_CHECK_EQUAL(parameter_types.id_of_satisfying_type(
            expression_types), 1u);

        expression_types = fl::ExpressionTypes({
            fern::ExpressionType(
                fern::DataTypes::STATIC_FIELD,
                fern::ValueTypes::FLOAT32),
            fern::ExpressionType(
                fern::DataTypes::CONSTANT,
                fern::ValueTypes::UINT8),
            fern::ExpressionType(
                fern::DataTypes::CONSTANT,
                fern::ValueTypes::FLOAT32)
        });
        BOOST_CHECK(parameter_types.is_satisfied_by(expression_types));
        BOOST_CHECK_EQUAL(parameter_types.id_of_satisfying_type(
            expression_types), 1u);
    }
    // constant, uint16
    {
        parameter_types = fl::ExpressionTypes({
            fern::ExpressionType(
                fern::DataTypes::CONSTANT,
                fern::ValueTypes::UINT16)
        });

        expression_types = fl::ExpressionTypes({
            fern::ExpressionType(
                fern::DataTypes::CONSTANT,
                fern::ValueTypes::UNSIGNED_INTEGER)
        });
        // Too broad. Expression properties are not a subset of the parameter's
        // properties. Casting (automatic or explicit) would help here.
        BOOST_CHECK(!parameter_types.is_satisfied_by(expression_types));
        BOOST_CHECK_EQUAL(parameter_types.id_of_satisfying_type(
            expression_types), 1u);
    }

    // constant, unsigned integer parameter
    {
        parameter_types = fl::ExpressionTypes({
            fern::ExpressionType(
                fern::DataTypes::STATIC_FIELD,
                fern::ValueTypes::FLOATING_POINT),
            fern::ExpressionType(
                fern::DataTypes::CONSTANT,
                fern::ValueTypes::UNSIGNED_INTEGER)
        });

        expression_types = fl::ExpressionTypes({
            fern::ExpressionType(
                fern::DataTypes::CONSTANT,
                fern::ValueTypes::UNSIGNED_INTEGER)
        });
        BOOST_CHECK(parameter_types.is_satisfied_by(expression_types));
        BOOST_CHECK_EQUAL(parameter_types.id_of_satisfying_type(
            expression_types), 0u);
    }
}

BOOST_AUTO_TEST_SUITE_END()
