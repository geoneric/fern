// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern algorithm algebra elementary multiply
#include <boost/test/unit_test.hpp>
#include "fern/core/data_customization_point/scalar.h"
#include "fern/core/type_traits.h"
#include "fern/core/types.h"
#include "fern/algorithm/core/test/test_utils.h"
#include "fern/algorithm/algebra/elementary/multiply.h"


namespace fa = fern::algorithm;


template<
    class Value1,
    class Value2,
    class Result>
using OutOfRangePolicy = fa::multiply::OutOfRangePolicy<Value1, Value2,
    Result>;


template<
    class Value1,
    class Value2,
    class Result>
struct VerifyWithinRange
{
    bool operator()(
        Value1 const& value1,
        Value2 const& value2)
    {
        fa::SequentialExecutionPolicy sequential;

        OutOfRangePolicy<Value1, Value2, Result> policy;
        Result result;

        fa::algebra::multiply(sequential, value1, value2, result);

        return policy.within_range(value1, value2, result);
    }
};


BOOST_AUTO_TEST_CASE(out_of_range_policy)
{
    // signed int * signed int
    {
        auto max_int32 = fern::max<int32_t>();
        auto min_int32 = fern::min<int32_t>();

        {
            VerifyWithinRange<int32_t, int32_t, int32_t> verify;
            BOOST_CHECK_EQUAL(verify(0, 0), true);
            BOOST_CHECK_EQUAL(verify(4, 5), true);
            BOOST_CHECK_EQUAL(verify(-4, 5), true);
            BOOST_CHECK_EQUAL(verify(-4, -5), true);

            // Both values positive.
            BOOST_CHECK_EQUAL(verify(2, max_int32), false);
            BOOST_CHECK_EQUAL(verify(max_int32, max_int32), false);

            // One value positive, one value negative.
            BOOST_CHECK_EQUAL(verify(2, -max_int32), false);
            BOOST_CHECK_EQUAL(verify(max_int32, -max_int32), false);

            // Both values negative.
            BOOST_CHECK_EQUAL(verify(-2, -max_int32), false);
            BOOST_CHECK_EQUAL(verify(-max_int32, -max_int32), false);
            BOOST_CHECK_EQUAL(verify(-2, min_int32), false);
            BOOST_CHECK_EQUAL(verify(min_int32, min_int32), false);
        }
    }

    // unsigned int * unsigned int
    {
        auto max_uint32 = fern::max<uint32_t>();

        {
            VerifyWithinRange<uint32_t, uint32_t, uint32_t> verify;
            BOOST_CHECK_EQUAL(verify(0, 0), true);
            BOOST_CHECK_EQUAL(verify(4, 5), true);
            BOOST_CHECK_EQUAL(verify(2, max_uint32), false);
            BOOST_CHECK_EQUAL(verify(max_uint32, max_uint32), false);
        }
    }

    // float * float
    {
        auto max_float32 = fern::max<fern::f32>();
        auto min_float32 = fern::min<fern::f32>();
        auto max_float64 = fern::max<fern::f64>();
        auto min_float64 = fern::min<fern::f64>();

        {
            VerifyWithinRange<fern::f32, fern::f32, fern::f32> verify;

            BOOST_CHECK_EQUAL(verify(0.0, 0.0), true);
            BOOST_CHECK_EQUAL(verify(4.0, 5.0), true);
            BOOST_CHECK_EQUAL(verify(-4.0, 5.0), true);
            BOOST_CHECK_EQUAL(verify(-4.0, -5.0), true);
            BOOST_CHECK_EQUAL(verify(max_float32, max_float32), false);
            BOOST_CHECK_EQUAL(verify(min_float32, min_float32), true);
        }

        {
            VerifyWithinRange<fern::f64, fern::f64, fern::f64> verify;

            BOOST_CHECK_EQUAL(verify(0.0, 0.0), true);
            BOOST_CHECK_EQUAL(verify(4.0, 5.0), true);
            BOOST_CHECK_EQUAL(verify(-4.0, 5.0), true);
            BOOST_CHECK_EQUAL(verify(-4.0, -5.0), true);
            BOOST_CHECK_EQUAL(verify(max_float64, max_float64), false);
            BOOST_CHECK_EQUAL(verify(min_float64, min_float64), true);
        }

        {
            VerifyWithinRange<fern::f32, fern::f64, fern::f64> verify;

            BOOST_CHECK_EQUAL(verify(0.0f, 0.0), true);
            BOOST_CHECK_EQUAL(verify(4.0f, 5.0), true);
            BOOST_CHECK_EQUAL(verify(-4.0f, 5.0), true);
            BOOST_CHECK_EQUAL(verify(-4.0f, -5.0), true);
            BOOST_CHECK_EQUAL(verify(max_float32, max_float64), false);
            BOOST_CHECK_EQUAL(verify(min_float32, min_float64), true);
        }
    }

    // unsigned int * signed int
    {
        auto max_int32 = fern::max<int32_t>();

        {
            VerifyWithinRange<uint32_t, int32_t, int64_t> verify;

            BOOST_CHECK_EQUAL(verify(0, 0), true);
            BOOST_CHECK_EQUAL(verify(4, 5), true);
            BOOST_CHECK_EQUAL(verify(4, -5), true);
            BOOST_CHECK_EQUAL(verify(2, max_int32), true);
        }
    }

    // unsigned int * float
    {
        auto max_float32 = fern::max<fern::f32>();

        {
            VerifyWithinRange<uint32_t, fern::f32, fern::f32> verify;

            BOOST_CHECK_EQUAL(verify(0, 0.0), true);
            BOOST_CHECK_EQUAL(verify(4, 5.0), true);
            BOOST_CHECK_EQUAL(verify(2, max_float32), false);
        }
    }

    // int * float
    {
        auto max_float32 = fern::max<fern::f32>();

        {
            VerifyWithinRange<int32_t, fern::f32, fern::f32> verify;

            BOOST_CHECK_EQUAL(verify(0, 0.0), true);
            BOOST_CHECK_EQUAL(verify(4, 5.0), true);
            BOOST_CHECK_EQUAL(verify(2, max_float32), false);
        }
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
    fa::SequentialExecutionPolicy sequential;

    Result result_we_get;
    fa::algebra::multiply(sequential, value1, value2, result_we_get);
    BOOST_CHECK_EQUAL(result_we_get, result_we_want);
}


BOOST_AUTO_TEST_CASE(algorithm)
{
    verify_value<int8_t, int8_t, int8_t>(-5, 6, -30);
    verify_value<int8_t, int8_t, int8_t>(-5, -6, 30);
    verify_value<uint8_t, uint8_t, uint8_t>(6, 5, 30);
    verify_value<double, float, double>(6.0, 5.0f, 30.0);
}
