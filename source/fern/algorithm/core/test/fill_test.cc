// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern algorithm core fill
#include <boost/test/unit_test.hpp>
#include "fern/core/data_customization_point/constant.h"
#include "fern/core/data_customization_point/vector.h"
#include "fern/algorithm/core/fill.h"
#include "fern/algorithm/statistic/count.h"


namespace fa = fern::algorithm;


BOOST_AUTO_TEST_SUITE(fill)

BOOST_AUTO_TEST_CASE(array_0d)
{
    int result = 5;
    fa::core::fill(fa::sequential, 3, result);
    BOOST_CHECK_EQUAL(result, 3);
}


BOOST_AUTO_TEST_CASE(array_1d)
{
    std::vector<int> result = { 1, 2, 3, 4, 5 };
    fa::core::fill(fa::sequential, 3, result);

    uint64_t count;
    fa::statistic::count(fa::sequential, result, 3, count);
    BOOST_CHECK_EQUAL(count, 5u);
}

BOOST_AUTO_TEST_SUITE_END()
