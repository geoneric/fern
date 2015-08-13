// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern algorithm algebra elementary trunc
#include <boost/test/unit_test.hpp>
#include "fern/core/data_customization_point/scalar.h"
#include "fern/algorithm/algebra/elementary/trunc.h"


namespace fa = fern::algorithm;


BOOST_AUTO_TEST_SUITE(trunc_)

template<
    class Value,
    class Result>
void verify_value(
    Value const& value,
    Result const& result_we_want)
{
    Result result_we_get;
    fa::algebra::trunc(fa::sequential, value, result_we_get);
    BOOST_CHECK_CLOSE(result_we_get, result_we_want, 1e-6);
}


BOOST_AUTO_TEST_CASE(algorithm)
{
    verify_value<double, double>( 0.0,  0.0);
    verify_value<double, double>( 0.1,  0.0);
    verify_value<double, double>( 0.5,  0.0);
    verify_value<double, double>(-0.1,  0.0);
    verify_value<double, double>(-0.5,  0.0);
}

BOOST_AUTO_TEST_SUITE_END()
