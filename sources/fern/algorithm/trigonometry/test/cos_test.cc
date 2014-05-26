#define BOOST_TEST_MODULE fern algorithm trigonometry cos
#include <boost/test/unit_test.hpp>
#include "fern/core/constant_traits.h"
#include "fern/core/type_traits.h"
#include "fern/algorithm/trigonometry/cos.h"


BOOST_AUTO_TEST_SUITE(cos_)

BOOST_AUTO_TEST_CASE(traits)
{
    using Cos = fern::trigonometry::Cos<float, float>;
    BOOST_CHECK((std::is_same<fern::OperationTraits<Cos>::category,
        fern::local_operation_tag>::value));
}


template<
    class Value>
using OutOfDomainPolicy = fern::cos::OutOfDomainPolicy<Value>;


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
using Algorithm = fern::cos::Algorithm<Value>;


template<
    class Value,
    class Result>
void verify_value(
    Value const& value,
    Result const& result_we_want)
{
    Result result_we_get;
    fern::trigonometry::cos(value, result_we_get);
    BOOST_CHECK_EQUAL(result_we_get, result_we_want);
}


BOOST_AUTO_TEST_CASE(algorithm)
{
    verify_value<double, double>(0.0, 1.0);
    verify_value<double, double>(fern::pi<double>(), -1.0);
    verify_value<double, double>(-fern::pi<double>(), -1.0);
}

BOOST_AUTO_TEST_SUITE_END()
