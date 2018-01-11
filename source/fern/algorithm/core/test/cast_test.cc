// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern algorithm core cast
#include <boost/test/unit_test.hpp>
#include "fern/core/data_customization_point/scalar.h"
#include "fern/core/types.h"
#include "fern/algorithm/core/cast.h"
#include "fern/algorithm/core/test/test_utils.h"


namespace fa = fern::algorithm;


template<
    class Value,
    class Result>
using OutOfRangePolicy = fa::cast::OutOfRangePolicy<Value, Result>;


template<
    class Value,
    class Result>
struct VerifyWithinRange
{
    bool operator()(
        Value const& value)
    {
        fa::SequentialExecutionPolicy sequential;

        OutOfRangePolicy<Value, Result> policy;
        Result result;
        fa::core::cast(sequential, value, result);
        return policy.within_range(value, result);
    }
};


BOOST_AUTO_TEST_CASE(out_of_range_policy)
{
    {
        auto max_int32 = fern::max<int32_t>();
        auto min_int32 = fern::min<int32_t>();
        VerifyWithinRange<int32_t, int32_t> verify;
        BOOST_CHECK_EQUAL(verify(5), true);
        BOOST_CHECK_EQUAL(verify(-5), true);
        BOOST_CHECK_EQUAL(verify(0), true);

        BOOST_CHECK_EQUAL(verify(min_int32), true);
        BOOST_CHECK_EQUAL(verify(max_int32), true);
    }

    {
        auto max_uint32 = fern::max<uint32_t>();
        auto min_uint32 = fern::min<uint32_t>();
        VerifyWithinRange<uint32_t, uint32_t> verify;
        BOOST_CHECK_EQUAL(verify(5), true);
        BOOST_CHECK_EQUAL(verify(0), true);

        BOOST_CHECK_EQUAL(verify(min_uint32), true);
        BOOST_CHECK_EQUAL(verify(max_uint32), true);
    }

    {
        auto max_float32 = fern::max<fern::f32>();
        auto min_float32 = fern::min<fern::f32>();
        VerifyWithinRange<fern::f32, fern::f32> verify;
        BOOST_CHECK_EQUAL(verify(5.5), true);
        BOOST_CHECK_EQUAL(verify(0.0), true);

        BOOST_CHECK_EQUAL(verify(min_float32), true);
        BOOST_CHECK_EQUAL(verify(max_float32), true);
    }
}


template<
    class Value,
    class Result>
void verify_value(
    Value const& value,
    Result const& result_we_want)
{
    fa::SequentialExecutionPolicy sequential;

    Result result_we_get;
    fa::core::cast(sequential, value, result_we_get);
    BOOST_CHECK_EQUAL(result_we_get, result_we_want);
}


BOOST_AUTO_TEST_CASE(algorithm)
{
    verify_value<int8_t, int8_t>( 5, 5);
    verify_value<int8_t, int8_t>(-5, -5);
    verify_value<int8_t, int8_t>( 0, 0);

    verify_value<bool, int8_t>(true, 1);
    verify_value<bool, int8_t>(false, 0);

    verify_value<uint8_t,uint8_t>(5, 5);
    verify_value<uint8_t,uint8_t>(0, 0);

    verify_value<bool, uint8_t>(true, 1);
    verify_value<bool, uint8_t>(false, 0);

    verify_value<double, double>( 5.5, 5.5);
    verify_value<double, double>(-5.5, -5.5);
    verify_value<double, double>( 0.0, 0.0);

    verify_value<bool, double>(true, 1.0);
    verify_value<bool, double>(false, 0.0);
}
