// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern algorithm algebra boole nor
#include <boost/test/unit_test.hpp>
#include "fern/core/data_customization_point/scalar.h"
#include "fern/core/type_traits.h"
#include "fern/algorithm/algebra/boole/nor.h"


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
    Result result_we_get;
    fa::algebra::nor(fa::sequential, value1, value2, result_we_get);
    BOOST_CHECK_EQUAL(result_we_get, result_we_want);
}


BOOST_AUTO_TEST_CASE(algorithm)
{
    verify_value<>(false, false, true);
    verify_value<>(false, true, false);
    verify_value<>(true, false, false);
    verify_value<>(true, true, false);

    verify_value<>(0, 0, 1);
    verify_value<>(0, 1, 0);
    verify_value<>(1, 0, 0);
    verify_value<>(1, 1, 0);

    verify_value<>( 0,  0,  1);
    verify_value<>( 0, -1,  0);
    verify_value<>(-1,  0,  0);
    verify_value<>(-1, -1,  0);

    verify_value<>( 0.0,  0.0,  1.0);
    verify_value<>( 0.0, -1.0,  0.0);
    verify_value<>(-1.0,  0.0,  0.0);
    verify_value<>(-1.0, -1.0,  0.0);
}
