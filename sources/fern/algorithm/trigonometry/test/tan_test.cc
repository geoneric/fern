#define BOOST_TEST_MODULE fern algorithm trigonometry tan
#include <boost/test/unit_test.hpp>
#include "fern/core/constant_traits.h"
#include "fern/algorithm/trigonometry/tan.h"


BOOST_AUTO_TEST_SUITE(tan_)

BOOST_AUTO_TEST_CASE(traits)
{
    using Tan = fern::trigonometry::Tan<float, float>;
    BOOST_CHECK((std::is_same<fern::OperationTraits<Tan>::category,
        fern::local_operation_tag>::value));
}


template<
    class Value>
using OutOfDomainPolicy = fern::tan::OutOfDomainPolicy<Value>;


BOOST_AUTO_TEST_CASE(out_of_domain_policy)
{
    {
        OutOfDomainPolicy<double> policy;
        BOOST_CHECK(policy.within_domain(5));
        BOOST_CHECK(policy.within_domain(-5));
        BOOST_CHECK(policy.within_domain(0));

        BOOST_CHECK(!policy.within_domain(fern::infinity<double>()));
        BOOST_CHECK(!policy.within_domain(-fern::infinity<double>()));

        BOOST_CHECK(!policy.within_domain(1.0 * fern::half_pi<double>()));
        BOOST_CHECK( policy.within_domain(2.0 * fern::half_pi<double>()));
        BOOST_CHECK(!policy.within_domain(3.0 * fern::half_pi<double>()));

        BOOST_CHECK(!policy.within_domain(-1.0 * fern::half_pi<double>()));
        BOOST_CHECK( policy.within_domain(-2.0 * fern::half_pi<double>()));
        BOOST_CHECK(!policy.within_domain(-3.0 * fern::half_pi<double>()));
    }
}


template<
    class Value,
    class Result>
using Algorithm = fern::tan::Algorithm<Value>;


template<
    class Value,
    class Result>
void verify_value(
    Value const& value,
    Result const& result_we_want)
{
    Result result_we_get;
    fern::trigonometry::tan(value, result_we_get);
    BOOST_CHECK_CLOSE(1.0 + result_we_get, 1.0 + result_we_want, 1e-10);
}


BOOST_AUTO_TEST_CASE(algorithm)
{
    verify_value<double, double>(0.0, 0.0);
    verify_value<double, double>(-0.0, -0.0);

    verify_value<double, double>(1.0 * fern::pi<double>(), 0.0);
    verify_value<double, double>(2.0 * fern::pi<double>(), 0.0);
    verify_value<double, double>(-1.0 * fern::pi<double>(), 0.0);
    verify_value<double, double>(-2.0 * fern::pi<double>(), 0.0);

    verify_value<double, double>(fern::half_pi<double>(),
        std::tan(fern::half_pi<double>()));
    verify_value<double, double>(-fern::half_pi<double>(),
        std::tan(-fern::half_pi<double>()));
}

BOOST_AUTO_TEST_SUITE_END()
