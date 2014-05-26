#define BOOST_TEST_MODULE fern algorithm trigonometry atan
#include <boost/test/unit_test.hpp>
#include "fern/core/constant_traits.h"
#include "fern/core/type_traits.h"
#include "fern/algorithm/trigonometry/atan.h"


BOOST_AUTO_TEST_SUITE(atan_)

BOOST_AUTO_TEST_CASE(traits)
{
    using ATan = fern::trigonometry::ATan<float, float>;
    BOOST_CHECK((std::is_same<fern::OperationTraits<ATan>::category,
        fern::local_operation_tag>::value));
}


template<
    class Value,
    class Result>
using Algorithm = fern::atan::Algorithm<Value>;


template<
    class Value,
    class Result>
void verify_value(
    Value const& value,
    Result const& result_we_want)
{
    Result result_we_get;
    fern::trigonometry::atan(value, result_we_get);
    BOOST_CHECK_EQUAL(result_we_get, result_we_want);
}


BOOST_AUTO_TEST_CASE(algorithm)
{
    verify_value<float, float>(0.0f, 0.0f);
    verify_value<float, float>(-0.0f, -0.0f);
    verify_value<float, float>(fern::infinity<float>(),
        fern::pi<float>() / 2.0f);
    verify_value<float, float>(-fern::infinity<float>(),
        -fern::pi<float>() / 2.0f);
}

BOOST_AUTO_TEST_SUITE_END()
