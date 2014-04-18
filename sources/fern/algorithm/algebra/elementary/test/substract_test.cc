#define BOOST_TEST_MODULE fern algorithm algebra elementary substract
#include <boost/test/unit_test.hpp>
#include "fern/core/constant_traits.h"
#include "fern/core/type_traits.h"
#include "fern/algorithm/algebra/elementary/substract.h"


BOOST_AUTO_TEST_SUITE(substract)

BOOST_AUTO_TEST_CASE(traits)
{
    using Substract = fern::algebra::Substract<int32_t, int32_t, int32_t>;
    BOOST_CHECK((std::is_same<fern::OperationTraits<Substract>::category,
        fern::local_operation_tag>::value));
}


template<
    class Value1,
    class Value2>
using OutOfDomainPolicy = fern::substract::OutOfDomainPolicy<Value1, Value2>;


BOOST_AUTO_TEST_CASE(out_of_domain_policy)
{
    {
        OutOfDomainPolicy<int32_t, int32_t> policy;
        BOOST_CHECK(policy.within_domain(5, 6));
        BOOST_CHECK(policy.within_domain(-5, -6));
        BOOST_CHECK(policy.within_domain(0, 0));
    }
}


template<
    class Value1,
    class Value2,
    class Result>
using OutOfRangePolicy = fern::substract::OutOfRangePolicy<Value1, Value2,
    Result>;


BOOST_AUTO_TEST_CASE(out_of_range_policy)
{
    auto max_int32 = fern::max<int32_t>();
    auto min_int32 = fern::min<int32_t>();
    auto max_int64 = fern::max<int64_t>();
    // auto min_int64 = fern::min<int64_t>();
    auto max_uint32 = fern::max<uint32_t>();
    auto min_uint32 = fern::min<uint32_t>();
    auto max_uint64 = fern::max<uint64_t>();
    // auto min_uint64 = fern::min<uint64_t>();
    // auto max_float32 = fern::max<float>();
    // auto min_float32 = fern::min<float>();
    auto max_float64 = fern::max<double>();
    // auto min_float64 = fern::min<double>();

    {
        // unsigned - unsigned
        OutOfRangePolicy<uint32_t, uint32_t, uint32_t> policy;
        BOOST_CHECK(!policy.within_range(5, 6, -1));
        BOOST_CHECK(policy.within_range(6, 5, 1));
        BOOST_CHECK(policy.within_range(0, 0, 0));
        BOOST_CHECK(policy.within_range(max_uint32, max_uint32,
            max_uint32 - max_uint32));
    }

    {
        // unsigned - signed
        {
            OutOfRangePolicy<uint32_t, int32_t, int64_t> policy;
            BOOST_CHECK(policy.within_range(5, 6, -1));
            BOOST_CHECK(policy.within_range(5, -6, 11));
            BOOST_CHECK(policy.within_range(0, 0, 0));
            BOOST_CHECK(policy.within_range(max_uint32, max_int32,
                static_cast<int64_t>(max_uint32) -
                static_cast<int64_t>(max_int32)));
            BOOST_CHECK(policy.within_range(max_uint32, -max_int32,
                static_cast<int64_t>(max_uint32) -
                static_cast<int64_t>(-max_int32)));
            BOOST_CHECK(policy.within_range(min_uint32, max_int32,
                static_cast<int64_t>(min_uint32) -
                static_cast<int64_t>(max_int32)));
            BOOST_CHECK(policy.within_range(min_uint32, -max_int32,
                static_cast<int64_t>(min_uint32) -
                static_cast<int64_t>(-max_int32)));
            BOOST_CHECK(policy.within_range(max_uint32, min_int32,
                static_cast<int64_t>(max_uint32) -
                static_cast<int64_t>(min_int32)));
            BOOST_CHECK(policy.within_range(max_uint32, -min_int32,
                static_cast<int64_t>(max_uint32) -
                static_cast<int64_t>(-min_int32)));
        }

        {
            OutOfRangePolicy<uint32_t, int64_t, int64_t> policy;
            BOOST_CHECK(policy.within_range(5, 6, -1));
            BOOST_CHECK(policy.within_range(5, -6, 11));
            BOOST_CHECK(policy.within_range(0, 0, 0));
            BOOST_CHECK(!policy.within_range(1, -max_int64,
                static_cast<int64_t>(1) - -max_int64));
        }

        {
            OutOfRangePolicy<uint64_t, int64_t, int64_t> policy;
            BOOST_CHECK(policy.within_range(5, 6, -1));
            BOOST_CHECK(policy.within_range(5, -6, 11));
            BOOST_CHECK(policy.within_range(0, 0, 0));
            BOOST_CHECK(!policy.within_range(1, -max_int64,
                static_cast<int64_t>(1) - -max_int64));
            BOOST_CHECK(policy.within_range(max_uint64, 0,
                static_cast<int64_t>(max_uint64) - static_cast<int64_t>(0)));
        }
    }

    {
        // signed - unsigned
        {
            OutOfRangePolicy<int32_t, uint32_t, int64_t> policy;
            BOOST_CHECK(policy.within_range(5, 6, -1));
            BOOST_CHECK(policy.within_range(-5, 6, -11));
            BOOST_CHECK(policy.within_range(0, 0, 0));
            BOOST_CHECK(policy.within_range(max_int32, max_uint32,
                static_cast<int64_t>(max_int32) -
                static_cast<int64_t>(max_uint32)));
            BOOST_CHECK(policy.within_range(max_int32, -max_uint32,
                static_cast<int64_t>(max_int32) -
                static_cast<int64_t>(-max_uint32)));
            BOOST_CHECK(policy.within_range(min_int32, max_uint32,
                static_cast<int64_t>(min_int32) -
                static_cast<int64_t>(max_uint32)));
            BOOST_CHECK(policy.within_range(min_int32, -max_uint32,
                static_cast<int64_t>(min_int32) -
                static_cast<int64_t>(-max_uint32)));
            BOOST_CHECK(policy.within_range(max_int32, min_uint32,
                static_cast<int64_t>(max_int32) -
                static_cast<int64_t>(min_uint32)));
            BOOST_CHECK(policy.within_range(max_int32, -min_uint32,
                static_cast<int64_t>(max_int32) -
                static_cast<int64_t>(-min_uint32)));
        }

        {
            OutOfRangePolicy<int64_t, uint32_t, int64_t> policy;
            BOOST_CHECK(policy.within_range(5, 6, -1));
            BOOST_CHECK(policy.within_range(-5, 6, -11));
            BOOST_CHECK(policy.within_range(0, 0, 0));

            // TODO

            // BOOST_CHECK(!policy.within_range(1, max_int64,
            //     static_cast<int64_t>(1) + max_int64));
        }

        // TODO
    }

    {
        // signed - signed
        OutOfRangePolicy<int32_t, int32_t, int32_t> policy;
        BOOST_CHECK(policy.within_range(5, 6, -1));
        BOOST_CHECK(policy.within_range(-5, -6, 1));
        BOOST_CHECK(policy.within_range(5, -6, 11));
        BOOST_CHECK(policy.within_range(0, 0, 0));
        BOOST_CHECK(!policy.within_range(max_int32, -1, max_int32 - -1));
        BOOST_CHECK(!policy.within_range(min_int32, 1, min_int32 - 1));
        BOOST_CHECK(!policy.within_range(1, min_int32, 1 - min_int32));
        BOOST_CHECK(!policy.within_range(-2, max_int32, -2 - max_int32));
    }

    {
        // float - float
        OutOfRangePolicy<double, double, double> policy;
        BOOST_CHECK(policy.within_range(5.0, 6.0, -1.0));
        BOOST_CHECK(policy.within_range(-5.0, -6.0, 1.0));
        BOOST_CHECK(policy.within_range(5.0, -6.0, 11.0));
        BOOST_CHECK(policy.within_range(0.0, 0.0, 0.0));
        BOOST_CHECK(!policy.within_range(-max_float64, max_float64,
            -max_float64 - max_float64));
        BOOST_CHECK(!policy.within_range(max_float64, -max_float64,
            max_float64 - -max_float64));
    }

    {
        // float - integer
        OutOfRangePolicy<double, int64_t, double> policy;
        BOOST_CHECK(policy.within_range(5.0, 6, -1.0));
        BOOST_CHECK(policy.within_range(-5.0, -6, 1.0));
        BOOST_CHECK(policy.within_range(5.0, -6, 11.0));
        BOOST_CHECK(policy.within_range(0.0, 0, 0.0));
        BOOST_CHECK(policy.within_range(-max_float64, max_int64,
            -max_float64 - max_int64));
        BOOST_CHECK(policy.within_range(max_float64, -max_int64,
            max_float64 - -max_int64));
    }

    {
        // integer - float
        OutOfRangePolicy<int64_t, double, double> policy;
        BOOST_CHECK(policy.within_range(5, 6.0, -1.0));
        BOOST_CHECK(policy.within_range(-5, -6.0, 1.0));
        BOOST_CHECK(policy.within_range(5, -6.0, 11.0));
        BOOST_CHECK(policy.within_range(0, 0.0, 0.0));
        BOOST_CHECK(policy.within_range(-max_int64, max_float64,
            -max_int64 - max_float64));
        BOOST_CHECK(policy.within_range(max_int64, -max_float64,
            max_int64 - -max_float64));
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
    Result result_we_get;
    fern::algebra::substract(value1, value2, result_we_get);
    BOOST_CHECK_EQUAL(result_we_get, result_we_want);
}


BOOST_AUTO_TEST_CASE(algorithm)
{
    verify_value<int8_t, int8_t, int8_t>(-5, 6, -11);
    verify_value<int8_t, int8_t, int8_t>(-5, -6, 1);
    verify_value<uint8_t, uint8_t, uint8_t>(6, 5, 1);
    verify_value<double, float, double>(6.0, 5.0f, 1.0);
}

BOOST_AUTO_TEST_SUITE_END()
