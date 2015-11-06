// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern feature core masked_scalar
#include <boost/test/unit_test.hpp>
#include "fern/feature/core/masked_scalar.h"


BOOST_AUTO_TEST_CASE(construct)
{
    // Default construct.
    {
        fern::MaskedScalar<int> masked_value;
        BOOST_CHECK(!masked_value.mask());
        BOOST_CHECK_EQUAL(masked_value.value(), 0);
        BOOST_CHECK_EQUAL(masked_value, 0);
    }

    // Construct with value.
    {
        fern::MaskedScalar<int> masked_value(5);
        BOOST_CHECK(!masked_value.mask());
        BOOST_CHECK_EQUAL(masked_value.value(), 5);
        BOOST_CHECK_EQUAL(masked_value, 5);
    }

    // Construct with value and mask.
    {
        fern::MaskedScalar<int> masked_value(5, false);
        BOOST_CHECK(!masked_value.mask());
        BOOST_CHECK_EQUAL(masked_value.value(), 5);
        BOOST_CHECK_EQUAL(masked_value, 5);
    }

    // Construct with value and mask.
    {
        fern::MaskedScalar<int> masked_value(5, true);
        BOOST_CHECK(masked_value.mask());
        BOOST_CHECK_EQUAL(masked_value.value(), 5);
        BOOST_CHECK_EQUAL(masked_value, 5);
    }
}


BOOST_AUTO_TEST_CASE(update)
{
    fern::MaskedScalar<int> masked_value(5);

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
    fern::MaskedScalar<int> value(5);
    value = 6;
    BOOST_CHECK(!value.mask());
    BOOST_CHECK_EQUAL(value, 6);

    value.mask() = true;

    value = 7;
    BOOST_CHECK(!value.mask());
    BOOST_CHECK_EQUAL(value, 7);
}
