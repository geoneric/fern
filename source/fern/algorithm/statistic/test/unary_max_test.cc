#define BOOST_TEST_MODULE fern algorithm statistic unary_max
#include <boost/test/unit_test.hpp>
#include "fern/core/constant_traits.h"
#include "fern/core/vector_traits.h"
#include "fern/feature/core/array_traits.h"
#include "fern/algorithm/statistic/unary_max.h"


namespace fa = fern::algorithm;


BOOST_AUTO_TEST_SUITE(unary_max)

template<
    class Value,
    class Result>
void verify_value(
    Value const& value,
    Result const& result_we_want)
{
    Result result_we_get;
    fa::statistic::unary_max(fa::sequential, value, result_we_get);
    BOOST_CHECK_EQUAL(result_we_get, result_we_want);
}


BOOST_AUTO_TEST_CASE(algorithm)
{
    // 0D array.
    {
        verify_value<int8_t, int8_t>( 5, 5);
        verify_value<int8_t, int8_t>(-5, -5);
        verify_value<int8_t, int8_t>( 0, 0);
    }

    // 1D array.
    {
        using Vector = std::vector<int32_t>;
        verify_value<Vector, int32_t>(Vector{2, 4, 6}, 6);
        verify_value<Vector, int32_t>(Vector{6, 4, 2}, 6);
    }

    // 2D array.
    {
        using Array = fern::Array<int32_t, 2>;
        verify_value<Array, int32_t>(Array{{2, 4, 6}}, 6);
        verify_value<Array, int32_t>(Array{{6, 4, 2}}, 6);
    }
}

BOOST_AUTO_TEST_SUITE_END()
