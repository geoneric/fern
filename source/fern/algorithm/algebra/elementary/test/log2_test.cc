// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern algorithm algebra elementary log2
#include <boost/test/unit_test.hpp>
#include "fern/core/data_customization_point/scalar.h"
#include "fern/algorithm/algebra/elementary/log2.h"


namespace fa = fern::algorithm;


template<
    class Value>
using OutOfDomainPolicy = fa::log2::OutOfDomainPolicy<Value>;


BOOST_AUTO_TEST_CASE(out_of_domain_policy)
{
    {
        OutOfDomainPolicy<double> policy;
        BOOST_CHECK( policy.within_domain( 5));
        BOOST_CHECK(!policy.within_domain(-5));
        BOOST_CHECK( policy.within_domain( 0));
        BOOST_CHECK( policy.within_domain(-0));
    }
}


template<
    class Value,
    class Result>
void verify_value(
    Value const& value,
    Result const& result_we_want)
{
    Result result_we_get;
    fa::algebra::log2(fa::sequential, value, result_we_get);
    BOOST_CHECK_EQUAL(result_we_get, result_we_want);
}


BOOST_AUTO_TEST_CASE(algorithm)
{
    verify_value<float, float>( 0.0f, -fern::infinity<float>());
    verify_value<float, float>( 1.0f, 0.0f);
    verify_value<float, float>( 9.0f, std::log2(9.0f));
}
