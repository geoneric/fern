// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern algorithm algebra boole defined
#include <boost/test/unit_test.hpp>
#include "fern/core/data_customization_point/constant.h"
#include "fern/core/type_traits.h"
#include "fern/feature/core/masked_constant.h"
#include "fern/algorithm/algebra/boole/defined.h"


namespace fa = fern::algorithm;


BOOST_AUTO_TEST_SUITE(defined)

template<
    class Value,
    class Result>
void verify_value(
    Value const& /* value */,
    Result const& result_we_want)
{
    Result result_we_get;
    fa::algebra::defined(fa::sequential, result_we_get);
    BOOST_CHECK_EQUAL(result_we_get, result_we_want);
}


BOOST_AUTO_TEST_CASE(algorithm)
{
    verify_value<bool, bool>(true, true);
    verify_value<bool, bool>(false, true);
}

BOOST_AUTO_TEST_SUITE_END()
