// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern algorithm accumulator sum
#include <boost/test/unit_test.hpp>
#include "fern/algorithm/algebra/elementary/add.h"
#include "fern/algorithm/accumulator/sum.h"


namespace fa = fern::algorithm;
namespace faa = fa::accumulator;

/// BOOST_AUTO_TEST_CASE(default_construct)
/// {
///     faa::Sum<int> sum;
///     BOOST_CHECK_EQUAL(sum(), 0);
/// }


/// BOOST_AUTO_TEST_CASE(accumulate)
/// {
///     faa::Sum<int> sum(5);
///     BOOST_CHECK_EQUAL(sum(), 5);
/// 
///     sum(2);
///     BOOST_CHECK_EQUAL(sum(), 7);
/// 
///     sum = 3;
///     BOOST_CHECK_EQUAL(sum(), 3);
/// }


BOOST_AUTO_TEST_CASE(out_of_range)
{
    faa::Sum<int> sum(std::numeric_limits<int>::max());
    BOOST_CHECK_EQUAL(sum(), std::numeric_limits<int>::max());

    BOOST_CHECK( sum.operator()<fa::add::OutOfRangePolicy>(0));
    BOOST_CHECK(!sum.operator()<fa::add::OutOfRangePolicy>(1));
}


/// BOOST_AUTO_TEST_CASE(merge)
/// {
///     auto sum(faa::Sum<int>(5) | faa::Sum<int>(6));
///     BOOST_CHECK_EQUAL(sum(), 11);
/// }
