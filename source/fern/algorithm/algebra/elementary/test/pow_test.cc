// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern algorithm algebra elementary pow
#include <boost/test/unit_test.hpp>
#include "fern/core/data_customization_point/scalar.h"
#include "fern/core/type_traits.h"
#include "fern/algorithm/algebra/elementary/pow.h"


namespace fa = fern::algorithm;


template<
    class Value1,
    class Value2>
using OutOfDomainPolicy = fa::pow::OutOfDomainPolicy<Value1, Value2>;


BOOST_AUTO_TEST_CASE(out_of_domain_policy)
{
    {
        OutOfDomainPolicy<double, double> policy;
        BOOST_CHECK(policy.within_domain(3.0, 4.0));
        BOOST_CHECK(policy.within_domain(0.0, 0.0));

        BOOST_CHECK(!policy.within_domain(-3.0, 4.1));
        BOOST_CHECK(!policy.within_domain(0.0, -1.0));
    }
}


template<
    class Value1,
    class Value2,
    class Result>
using OutOfRangePolicy = fa::pow::OutOfRangePolicy<Value1, Value2, Result>;


template<
    class Value1,
    class Value2,
    class Result>
struct VerifyWithinRange
{
    bool operator()(
        Value1 const& value1,
        Value2 const& value2)
    {
        fa::SequentialExecutionPolicy sequential;

        OutOfRangePolicy<Value1, Value2, Result> policy;
        Result result;

        fa::algebra::pow(sequential, value1, value2, result);

        return policy.within_range(value1, value2, result);
    }
};


BOOST_AUTO_TEST_CASE(out_of_range_policy)
{
    {
        float max_float = fern::max<float>();

        VerifyWithinRange<float, float, float> verify;
        BOOST_CHECK_EQUAL(verify(0.0, 0.0), true);

        BOOST_CHECK_EQUAL(verify(max_float, max_float), false);
    }
}


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
    fa::algebra::pow(sequential, value1, value2, result_we_get);
    BOOST_CHECK_EQUAL(result_we_get, result_we_want);
}


BOOST_AUTO_TEST_CASE(algorithm)
{
    verify_value<float, float, float>( 0.0f, 0.0f, 1.0f);
    verify_value<float, float, float>( 1.0f, 1.0f, 1.0f);
    verify_value<float, float, float>( 2.0f, 2.0f, 4.0f);
}
