#define BOOST_TEST_MODULE fern algorithm algebra elementary sqrt
#include <boost/test/unit_test.hpp>
#include "fern/core/constant_traits.h"
#include "fern/algorithm/algebra/elementary/sqrt.h"


BOOST_AUTO_TEST_SUITE(sqrt_)

BOOST_AUTO_TEST_CASE(traits)
{
    using Sqrt = fern::algebra::Sqrt<float, float>;
    BOOST_CHECK((std::is_same<fern::OperationTraits<Sqrt>::category,
        fern::local_operation_tag>::value));
}


template<
    class Value>
using OutOfDomainPolicy = fern::sqrt::OutOfDomainPolicy<Value>;


BOOST_AUTO_TEST_CASE(out_of_domain_policy)
{
    {
        OutOfDomainPolicy<double> policy;
        BOOST_CHECK(policy.within_domain(5));
        BOOST_CHECK(!policy.within_domain(-5));
        BOOST_CHECK(policy.within_domain(0));
        BOOST_CHECK(policy.within_domain(-0));
    }
}


template<
    class Value,
    class Result>
using Algorithm = fern::sqrt::Algorithm<Value>;


template<
    class Value,
    class Result>
void verify_value(
    Value const& value,
    Result const& result_we_want)
{
    Result result_we_get;
    fern::algebra::sqrt(value, result_we_get);
    BOOST_CHECK_EQUAL(result_we_get, result_we_want);
}


BOOST_AUTO_TEST_CASE(algorithm)
{
    verify_value<float, float>( 0.0f, 0.0f);
    verify_value<float, float>( 1.0f, 1.0f);
    verify_value<float, float>( 9.0f, 3.0f);
}

BOOST_AUTO_TEST_SUITE_END()
