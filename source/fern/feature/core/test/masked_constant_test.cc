#define BOOST_TEST_MODULE fern feature core
#include <boost/test/unit_test.hpp>
#include "fern/feature/core/masked_constant.h"


BOOST_AUTO_TEST_SUITE(masked_constant)

BOOST_AUTO_TEST_CASE(construct)
{
    // Default construct.
    {
        fern::MaskedConstant<int> masked_value;
        BOOST_CHECK(!masked_value.mask());
        BOOST_CHECK_EQUAL(masked_value.value(), 0);
        BOOST_CHECK_EQUAL(masked_value, 0);
    }

    // Construct with value.
    {
        fern::MaskedConstant<int> masked_value(5);
        BOOST_CHECK(!masked_value.mask());
        BOOST_CHECK_EQUAL(masked_value.value(), 5);
        BOOST_CHECK_EQUAL(masked_value, 5);
    }

    // Construct with value and mask.
    {
        fern::MaskedConstant<int> masked_value(5, false);
        BOOST_CHECK(!masked_value.mask());
        BOOST_CHECK_EQUAL(masked_value.value(), 5);
        BOOST_CHECK_EQUAL(masked_value, 5);
    }

    // Construct with value and mask.
    {
        fern::MaskedConstant<int> masked_value(5, true);
        BOOST_CHECK(masked_value.mask());
        BOOST_CHECK_EQUAL(masked_value.value(), 5);
        BOOST_CHECK_EQUAL(masked_value, 5);
    }
}


BOOST_AUTO_TEST_CASE(update)
{
    fern::MaskedConstant<int> masked_value(5);

    BOOST_CHECK(!masked_value.mask());
    BOOST_CHECK_EQUAL(masked_value.value(), 5);
    BOOST_CHECK_EQUAL(masked_value, 5);

    masked_value.value() = 6;
    BOOST_CHECK(!masked_value.mask());
    BOOST_CHECK_EQUAL(masked_value.value(), 6);
    BOOST_CHECK_EQUAL(masked_value, 6);

    masked_value.mask() = true;
    BOOST_CHECK(masked_value.mask());
    BOOST_CHECK_EQUAL(masked_value.value(), 6);
    BOOST_CHECK_EQUAL(masked_value, 6);
}


BOOST_AUTO_TEST_CASE(assign)
{
    fern::MaskedConstant<int> value(5);
    value = 6;
    BOOST_CHECK(!value.mask());
    BOOST_CHECK_EQUAL(value, 6);

    value.mask() = true;

    value = 7;
    BOOST_CHECK(!value.mask());
    BOOST_CHECK_EQUAL(value, 7);
}

BOOST_AUTO_TEST_SUITE_END()
