// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern algorithm algebra elementary factorial
#include <boost/test/unit_test.hpp>
#include "fern/core/data_customization_point/scalar.h"
#include "fern/algorithm/core/test/test_utils.h"
#include "fern/algorithm/algebra/elementary/factorial.h"


namespace fa = fern::algorithm;


template<
    class Value>
using OutOfDomainPolicy = fa::factorial::OutOfDomainPolicy<Value>;


BOOST_AUTO_TEST_CASE(out_of_domain_policy)
{
    {
        OutOfDomainPolicy<int> policy;
        BOOST_CHECK( policy.within_domain(5));
        BOOST_CHECK(!policy.within_domain(-5));
        BOOST_CHECK( policy.within_domain(0));
        BOOST_CHECK( policy.within_domain(-0));
    }

    {
        OutOfDomainPolicy<double> policy;
        BOOST_CHECK( policy.within_domain(5));
        BOOST_CHECK(!policy.within_domain(-5));
        BOOST_CHECK(!policy.within_domain(-5.01));
        BOOST_CHECK(!policy.within_domain(-4.99));
        BOOST_CHECK( policy.within_domain(0));
        BOOST_CHECK( policy.within_domain(-0));
    }
}


template<
    class Value,
    class Result>
using OutOfRangePolicy = fa::factorial::OutOfRangePolicy<Value, Result>;


template<
    class Value,
    class Result>
struct VerifyWithinRange
{
    bool operator()(
        Value const& value)
    {
        OutOfRangePolicy<Value, Result> policy;
        Result result;

        fa::algebra::factorial(fa::sequential, value, result);

        return policy.within_range(value, result);
    }
};


BOOST_AUTO_TEST_CASE(out_of_range_policy)
{
    {
        VerifyWithinRange<uint8_t, uint8_t> verify;
        BOOST_CHECK_EQUAL(verify(0), true);
        BOOST_CHECK_EQUAL(verify(5), true);
        BOOST_CHECK_EQUAL(verify(6), false);
    }

    {
        VerifyWithinRange<int32_t, int32_t> verify;
        BOOST_CHECK_EQUAL(verify(0), true);
        BOOST_CHECK_EQUAL(verify(5), true);
        BOOST_CHECK_EQUAL(verify(6), true);
        BOOST_CHECK_EQUAL(verify(11), true);
        BOOST_CHECK_EQUAL(verify(13), false);
    }

    {
        VerifyWithinRange<uint32_t, uint32_t> verify;
        BOOST_CHECK_EQUAL(verify(0), true);
        BOOST_CHECK_EQUAL(verify(5), true);
        BOOST_CHECK_EQUAL(verify(11), true);
        BOOST_CHECK_EQUAL(verify(13), false);
    }

    {
        VerifyWithinRange<uint64_t, uint64_t> verify;
        BOOST_CHECK_EQUAL(verify(0), true);
        BOOST_CHECK_EQUAL(verify(5), true);
        BOOST_CHECK_EQUAL(verify(20), true);
        BOOST_CHECK_EQUAL(verify(21), false);
    }

    {
        VerifyWithinRange<float, float> verify;
        BOOST_CHECK_EQUAL(verify(34.0f), true);
        BOOST_CHECK_EQUAL(verify(35.0f), false);
        BOOST_CHECK_EQUAL(verify(fern::infinity<float>()), false);
    }

    {
        VerifyWithinRange<double, double> verify;
        BOOST_CHECK_EQUAL(verify(170.0), true);
        BOOST_CHECK_EQUAL(verify(171.0), false);
        BOOST_CHECK_EQUAL(verify(fern::infinity<double>()), false);
    }
}


template<
    class Value,
    class Result>
void verify_integral_value(
    Value const& value,
    Result const& result_we_want)
{
    Result result_we_get;
    fa::algebra::factorial(fa::sequential, value, result_we_get);

    BOOST_CHECK_EQUAL(result_we_get, result_we_want);
}


template<
    class Value,
    class Result>
void verify_floating_point_value(
    Value const& value,
    Result const& result_we_want)
{
    Result result_we_get;
    fa::algebra::factorial(fa::sequential, value, result_we_get);

    BOOST_CHECK_CLOSE(result_we_get, result_we_want, 1e-6);
}


BOOST_AUTO_TEST_CASE(algorithm)
{
    {
        verify_integral_value<uint8_t, uint8_t>(0, 1);
        verify_integral_value<uint8_t, uint8_t>(1, 1);
        verify_integral_value<uint8_t, uint8_t>(2, 2);
        verify_integral_value<uint8_t, uint8_t>(3, 6);
        verify_integral_value<uint8_t, uint8_t>(5, 120);
    }

    {
        verify_integral_value<int32_t, int32_t>(0, 1);
        verify_integral_value<int32_t, int32_t>(1, 1);
        verify_integral_value<int32_t, int32_t>(2, 2);
        verify_integral_value<int32_t, int32_t>(3, 6);
        verify_integral_value<int32_t, int32_t>(6, 720);
        verify_integral_value<int32_t, int32_t>(12, 479001600);
    }

    {
        verify_floating_point_value<double, double>(0.0, 1.0);
        verify_floating_point_value<double, double>(1.0, 1.0);
        verify_floating_point_value<double, double>(10.0, 3628800.0);
    }
}
