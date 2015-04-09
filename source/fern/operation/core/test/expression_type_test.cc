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
