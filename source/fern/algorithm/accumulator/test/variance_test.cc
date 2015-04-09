// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern algorithm accumulator
#include <boost/test/unit_test.hpp>
#include "fern/algorithm/accumulator/variance.h"


namespace faa = fern::algorithm::accumulator;


// TODO Once we have a full blown square algorithm we must use that
//      instead of this one.
template<
    typename T>
inline constexpr T square(
    T const& value)
{
    return value * value;
}


BOOST_AUTO_TEST_SUITE(variance)

BOOST_AUTO_TEST_CASE(default_construct)
{
    faa::Variance<int> variance;
}


BOOST_AUTO_TEST_CASE(accumulate)
{
    {
        faa::Variance<int> variance(5);
        // 5
        BOOST_CHECK_EQUAL(variance(), (
            square(5 - 5)) / 1);

        variance(2);
        // 5 2
        BOOST_CHECK_EQUAL(variance(), (
            square(3 - 5) +
            square(3 - 2)) / 2);

        variance(3);
        // 5 2 3
        BOOST_CHECK_EQUAL(variance(), (
            square(3 - 5) +
            square(3 - 2) +
            square(3 - 3)) / 3);

        variance = 8;
        // 8
        BOOST_CHECK_EQUAL(variance(), (
            square(8 - 8)) / 1);
    }

    {
        faa::Variance<int, double> variance(5);
        // 5
        BOOST_CHECK_EQUAL(variance(), (
            square(5.0 - 5.0)) / 1.0);

        variance(2);
        // 5 2
        BOOST_CHECK_EQUAL(variance(), (
            square(3.5 - 5.0) +
            square(3.5 - 2.0)) / 2.0);

        variance = 3;
        // 3
        BOOST_CHECK_EQUAL(variance(), (
            square(3.0 - 3.0)) / 1.0);
    }
}


BOOST_AUTO_TEST_CASE(merge)
{
    {
        auto variance(faa::Variance<int>(15) | faa::Variance<int>(5));
        // 15 5
        BOOST_CHECK_EQUAL(variance(), (
            square(10 - 15) +
            square(10 - 5)) / 2);
    }

    {
        auto variance(faa::Variance<int, double>(5) |
            faa::Variance<int, double>(20));
        // 5 20
        BOOST_CHECK_EQUAL(variance(), (
            square(12.5 - 5.0) +
            square(12.5 - 20.0)) / 2.0);
    }
}

BOOST_AUTO_TEST_SUITE_END()
