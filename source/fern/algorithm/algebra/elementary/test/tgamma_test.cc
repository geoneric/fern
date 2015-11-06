// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern algorithm algebra elementary tgamma
#include <boost/test/unit_test.hpp>
#include "fern/core/data_customization_point/scalar.h"
#include "fern/algorithm/algebra/elementary/tgamma.h"


namespace fa = fern::algorithm;


template<
    class Value>
using OutOfDomainPolicy = fa::tgamma::OutOfDomainPolicy<Value>;


BOOST_AUTO_TEST_CASE(out_of_domain_policy)
{
    {
        OutOfDomainPolicy<double> policy;
        BOOST_CHECK( policy.within_domain(5));
        BOOST_CHECK(!policy.within_domain(-5));
        BOOST_CHECK( policy.within_domain(-5.01));
        BOOST_CHECK( policy.within_domain(-4.99));
        BOOST_CHECK(policy.within_domain(0));
        BOOST_CHECK(policy.within_domain(-0));
    }
}


template<
    class Value,
    class Result>
using OutOfRangePolicy = fa::tgamma::OutOfRangePolicy<Value, Result>;


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

        fa::algebra::tgamma(fa::sequential, value, result);

        return policy.within_range(value, result);
    }
};


BOOST_AUTO_TEST_CASE(out_of_range_policy)
{
    {
        VerifyWithinRange<double, double> verify;
        BOOST_CHECK_EQUAL(verify(-1.0), false);
        BOOST_CHECK_EQUAL(verify(fern::infinity<double>()), false);
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
    fa::algebra::tgamma(fa::sequential, value, result_we_get);
    BOOST_CHECK_CLOSE(result_we_get, result_we_want, 1e-6);
}


BOOST_AUTO_TEST_CASE(algorithm)
{
    verify_value<double, double>(10.0, 362880.0);
    verify_value<double, double>(0.5, std::sqrt(fern::pi<double>()));
    verify_value<double, double>(1.0, 1.0);
}
