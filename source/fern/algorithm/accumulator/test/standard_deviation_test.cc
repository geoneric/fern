// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern algorithm accumulator standard_deviation
#include <boost/test/unit_test.hpp>
#include "fern/algorithm/accumulator/standard_deviation.h"


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


BOOST_AUTO_TEST_CASE(default_construct)
{
    faa::StandardDeviation<int> standard_deviation;
}


BOOST_AUTO_TEST_CASE(accumulate)
{
    {
        faa::StandardDeviation<int> standard_deviation(5);
        // 5
        BOOST_CHECK_EQUAL(standard_deviation(), static_cast<int>(std::sqrt((
            square(5 - 5)) / 1)));

        standard_deviation(2);
        // 5 2
        BOOST_CHECK_EQUAL(standard_deviation(), static_cast<int>(std::sqrt((
            square(3 - 5) +
            square(3 - 2)) / 2)));

        standard_deviation(3);
        // 5 2 3
        BOOST_CHECK_EQUAL(standard_deviation(), static_cast<int>(std::sqrt((
            square(3 - 5) +
            square(3 - 2) +
            square(3 - 3)) / 3)));

        standard_deviation = 8;
        // 8
        BOOST_CHECK_EQUAL(standard_deviation(), static_cast<int>(std::sqrt((
            square(8 - 8)) / 1)));
    }

    {
        faa::StandardDeviation<int, double> standard_deviation(5);
        // 5
        BOOST_CHECK_EQUAL(standard_deviation(), std::sqrt((
            square(5.0 - 5.0)) / 1.0));

        standard_deviation(2);
        // 5 2
        BOOST_CHECK_EQUAL(standard_deviation(), std::sqrt((
            square(3.5 - 5.0) +
            square(3.5 - 2.0)) / 2.0));

        standard_deviation = 3;
        // 3
        BOOST_CHECK_EQUAL(standard_deviation(), std::sqrt((
            square(3.0 - 3.0)) / 1.0));
    }
}


BOOST_AUTO_TEST_CASE(merge)
{
    {
        auto standard_deviation(faa::StandardDeviation<int>(15) |
            faa::StandardDeviation<int>(5));
        // 15 5
        BOOST_CHECK_EQUAL(standard_deviation(), static_cast<int>(std::sqrt((
            square(10 - 15) +
            square(10 - 5)) / 2)));
    }

    {
        auto standard_deviation(faa::StandardDeviation<int, double>(5) |
            faa::StandardDeviation<int, double>(20));
        // 5 20
        BOOST_CHECK_EQUAL(standard_deviation(), std::sqrt((
            square(12.5 - 5.0) +
            square(12.5 - 20.0)) / 2.0));
    }
}
