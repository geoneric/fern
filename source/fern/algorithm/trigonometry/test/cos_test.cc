#define BOOST_TEST_MODULE fern algorithm trigonometry cos
#include <boost/test/unit_test.hpp>
#include "fern/core/data_customization_point/constant.h"
#include "fern/core/type_traits.h"
#include "fern/algorithm/trigonometry/cos.h"


namespace fa = fern::algorithm;


BOOST_AUTO_TEST_SUITE(cos_)

template<
    class Value>
using OutOfDomainPolicy = fa::cos::OutOfDomainPolicy<Value>;


BOOST_AUTO_TEST_CASE(out_of_domain_policy)
{
    {
        OutOfDomainPolicy<double> policy;
        BOOST_CHECK(policy.within_domain(5));
        BOOST_CHECK(policy.within_domain(-5));
        BOOST_CHECK(policy.within_domain(0));
        BOOST_CHECK(!policy.within_domain(fern::infinity<double>()));
        BOOST_CHECK(!policy.within_domain(-fern::infinity<double>()));
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
    fa::trigonometry::cos(fa::sequential, value, result_we_get);
    BOOST_CHECK_EQUAL(result_we_get, result_we_want);
}


BOOST_AUTO_TEST_CASE(algorithm)
{
    verify_value<double, double>(0.0, 1.0);
    verify_value<double, double>(fern::pi<double>(), -1.0);
    verify_value<double, double>(-fern::pi<double>(), -1.0);
}

BOOST_AUTO_TEST_SUITE_END()
