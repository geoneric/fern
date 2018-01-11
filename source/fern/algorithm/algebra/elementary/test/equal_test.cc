// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern algorithm algebra elementary equal
#include <boost/test/unit_test.hpp>
#include "fern/core/data_customization_point/scalar.h"
#include "fern/algorithm/algebra/elementary/equal.h"


namespace fa = fern::algorithm;


template<
    class Value1,
    class Value2,
    class Result>
void verify_value(
    Value1 const& value1,
    Value2 const& value2,
    Result const& result_we_want)
{
    fa::SequentialExecutionPolicy sequential;

    Result result_we_get;
    fa::algebra::equal(sequential, value1, value2, result_we_get);
    BOOST_CHECK_EQUAL(result_we_get, result_we_want);
}


BOOST_AUTO_TEST_CASE(algorithm)
{
    verify_value<float, float, bool>(0.0f, 0.0f, true);
    verify_value<float, float, bool>(1.0f, 1.0f, true);
    verify_value<float, float, bool>(-1.0f, -1.0f, true);

    verify_value<int, int, bool>(1, 2, false);
    verify_value<int, int, bool>(2, 1, false);

    verify_value<int, int, bool>(-1, -2, false);
    verify_value<int, int, bool>(-2, -1, false);
}
