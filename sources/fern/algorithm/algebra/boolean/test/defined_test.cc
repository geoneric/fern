#define BOOST_TEST_MODULE fern algorithm algebra boolean defined
#include <boost/test/unit_test.hpp>
#include "fern/core/constant_traits.h"
#include "fern/core/type_traits.h"
#include "fern/feature/core/masked_constant.h"
#include "fern/algorithm/algebra/boolean/defined.h"


BOOST_AUTO_TEST_SUITE(defined)

BOOST_AUTO_TEST_CASE(traits)
{
    using Defined = fern::algebra::Defined<int32_t, bool>;
    BOOST_CHECK((std::is_same<fern::OperationTraits<Defined>::category,
        fern::local_operation_tag>::value));
}


template<
    class Value>
using OutOfDomainPolicy = fern::defined::OutOfDomainPolicy<Value>;


BOOST_AUTO_TEST_CASE(out_of_domain_policy)
{
    {
        OutOfDomainPolicy<int32_t> policy;
        BOOST_CHECK(policy.within_domain(5));
        BOOST_CHECK(policy.within_domain(-5));
        BOOST_CHECK(policy.within_domain(0));
    }
}


template<
    class Value,
    class Result>
using OutOfRangePolicy = fern::defined::OutOfRangePolicy<Value, Result>;


template<
    class Value,
    class Result>
using Algorithm = fern::defined::Algorithm<Value>;


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
        // using Constant = fern::MaskedConstant<int32_t>;
        // VerifyWithinRange<Constant, bool> verify;

        // BOOST_CHECK_EQUAL(verify(Constant(5, true)), true);
        // BOOST_CHECK_EQUAL(verify(Constant(-5, true)), true);
        // BOOST_CHECK_EQUAL(verify(Constant(0, true)), true);
        // BOOST_CHECK_EQUAL(verify(Constant(5, false)), true);
        // BOOST_CHECK_EQUAL(verify(Constant(-5, false)), true);
        // BOOST_CHECK_EQUAL(verify(Constant(0, false)), true);

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
    fern::algebra::defined(value, result_we_get);
    BOOST_CHECK_EQUAL(result_we_get, result_we_want);
}


BOOST_AUTO_TEST_CASE(algorithm)
{
    verify_value<bool, bool>(true, true);
    verify_value<bool, bool>(false, false);
}

BOOST_AUTO_TEST_SUITE_END()
