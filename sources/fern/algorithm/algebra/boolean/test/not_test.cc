#define BOOST_TEST_MODULE fern algorithm algebra boolean not
#include <boost/test/unit_test.hpp>
#include "fern/core/constant_traits.h"
#include "fern/core/type_traits.h"
#include "fern/feature/core/masked_constant.h"
#include "fern/algorithm/algebra/boolean/not.h"


BOOST_AUTO_TEST_SUITE(not_)

BOOST_AUTO_TEST_CASE(traits)
{
    using Not = fern::algebra::Not<bool, bool>;
    BOOST_CHECK((std::is_same<fern::OperationTraits<Not>::category,
        fern::local_operation_tag>::value));
}


template<
    class Value>
using OutOfDomainPolicy = fern::not_::OutOfDomainPolicy<Value>;


BOOST_AUTO_TEST_CASE(out_of_domain_policy)
{
    {
        OutOfDomainPolicy<bool> policy;
        BOOST_CHECK(policy.within_domain(5));
        BOOST_CHECK(policy.within_domain(-5));
        BOOST_CHECK(policy.within_domain(0));
    }
}


template<
    class Value,
    class Result>
using OutOfRangePolicy = fern::not_::OutOfRangePolicy<Value, Result>;


template<
    class Value,
    class Result>
using Algorithm = fern::not_::Algorithm<Value>;


template<
    class Value,
    class Result>
struct VerifyWithinRange
{
    bool operator()(
        Value const& value)
    {
        OutOfRangePolicy<Value, Result> policy;
        Algorithm<Value, Result> algorithm;
        Result result;
        algorithm(value, result);
        return policy.within_range(value, result);
    }
};


BOOST_AUTO_TEST_CASE(out_of_range_policy)
{
    {
        VerifyWithinRange<bool, bool> verify;

        BOOST_CHECK_EQUAL(verify(true), true);
        BOOST_CHECK_EQUAL(verify(false), true);
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
    fern::algebra::not_(value, result_we_get);
    BOOST_CHECK_EQUAL(result_we_get, result_we_want);
}


BOOST_AUTO_TEST_CASE(algorithm)
{
    verify_value<bool, bool>(true, false);
    verify_value<bool, bool>(false, true);
}

BOOST_AUTO_TEST_SUITE_END()
