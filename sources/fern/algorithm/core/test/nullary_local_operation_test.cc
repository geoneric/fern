#define BOOST_TEST_MODULE fern algorithm algebra nullary_local_operation
#include <boost/test/unit_test.hpp>
#include "fern/core/constant_traits.h"
#include "fern/core/vector_traits.h"
#include "fern/feature/core/array_traits.h"
#include "fern/feature/core/masked_array_traits.h"
#include "fern/feature/core/masked_constant_traits.h"
#include "fern/algorithm/algebra/boolean/defined.h"


using ArgumentValue = int32_t;
using ResultValue = int32_t;
using Algorithm = fern::defined::detail::Algorithm;


BOOST_AUTO_TEST_SUITE(nullary_local_operation)

BOOST_AUTO_TEST_CASE(array_0d)
{
    using InputNoDataPolicy = fern::SkipNoData;
    using OutputNoDataPolicy = fern::DontMarkNoData;
    using Result = ResultValue;

    OutputNoDataPolicy output_no_data_policy;
    Result result{3};

    fern::nullary_local_operation<Algorithm>(
        InputNoDataPolicy(), output_no_data_policy,
        fern::sequential, result);

    BOOST_REQUIRE_EQUAL(result, 1);

    result = 3;

    fern::nullary_local_operation<Algorithm>(
        InputNoDataPolicy(), output_no_data_policy,
        fern::parallel, result);

    BOOST_REQUIRE_EQUAL(result, 1);
}


BOOST_AUTO_TEST_CASE(array_0d_masked)
{
    using InputNoDataPolicy = fern::DetectNoDataByValue<bool>;
    using OutputNoDataPolicy = fern::MarkNoDataByValue<bool>;
    using Result = fern::MaskedConstant<ResultValue>;

    Result result;

    // Input is not masked.
    {
        result.value() = 3;
        result.mask() = false;

        InputNoDataPolicy input_no_data_policy(result.mask(), true);
        OutputNoDataPolicy output_no_data_policy(result.mask(), true);

        fern::nullary_local_operation<Algorithm>(
            input_no_data_policy, output_no_data_policy,
            fern::sequential, result);

        BOOST_REQUIRE_EQUAL(result.mask(), false);
        BOOST_REQUIRE_EQUAL(result.value(), 1);
    }

    // Input is masked.
    {
        result.value() = 3;
        result.mask() = true;

        InputNoDataPolicy input_no_data_policy(result.mask(), true);
        OutputNoDataPolicy output_no_data_policy(result.mask(), true);
        fern::nullary_local_operation<Algorithm>(
            input_no_data_policy, output_no_data_policy,
            fern::sequential, result);

        BOOST_REQUIRE_EQUAL(result.mask(), true);
        BOOST_REQUIRE_EQUAL(result.value(), 3);
    }
}


BOOST_AUTO_TEST_CASE(array_1d_sequential)
{
    using InputNoDataPolicy = fern::SkipNoData;
    using OutputNoDataPolicy = fern::DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;

    // vector
    {
        using Result = std::vector<ResultValue>;

        Result result{3, 3, 3};

        fern::nullary_local_operation<Algorithm>(
            InputNoDataPolicy(), output_no_data_policy,
            fern::sequential, result);

        BOOST_REQUIRE_EQUAL(result[0], 1);
        BOOST_REQUIRE_EQUAL(result[1], 1);
        BOOST_REQUIRE_EQUAL(result[2], 1);
    }

    // 1d array
    {
        using Result = fern::Array<ResultValue, 1>;

        Result result{3, 3, 3};

        fern::nullary_local_operation<Algorithm>(
            InputNoDataPolicy(), output_no_data_policy,
            fern::sequential, result);

        BOOST_REQUIRE_EQUAL(result[0], 1);
        BOOST_REQUIRE_EQUAL(result[1], 1);
        BOOST_REQUIRE_EQUAL(result[2], 1);
    }

    // empty
    {
        using Result = std::vector<ResultValue>;

        Result result;

        fern::nullary_local_operation<Algorithm>(
            InputNoDataPolicy(), output_no_data_policy,
            fern::sequential, result);

        BOOST_CHECK(result.empty());
    }
}


BOOST_AUTO_TEST_CASE(array_1d_parallel)
{
    using InputNoDataPolicy = fern::SkipNoData;
    using OutputNoDataPolicy = fern::DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;

    fern::ThreadClient client(2);

    // vector
    {
        using Result = std::vector<ResultValue>;

        Result result{3, 3, 3};

        fern::nullary_local_operation<Algorithm>(
            InputNoDataPolicy(), output_no_data_policy,
            fern::parallel, result);

        BOOST_REQUIRE_EQUAL(result[0], 1);
        BOOST_REQUIRE_EQUAL(result[1], 1);
        BOOST_REQUIRE_EQUAL(result[2], 1);
    }

    // 1d array
    {
        using Result = fern::Array<ResultValue, 1>;

        Result result{3, 3, 3};

        fern::nullary_local_operation<Algorithm>(
            InputNoDataPolicy(), output_no_data_policy,
            fern::parallel, result);

        BOOST_REQUIRE_EQUAL(result[0], 1);
        BOOST_REQUIRE_EQUAL(result[1], 1);
        BOOST_REQUIRE_EQUAL(result[2], 1);
    }

    // empty
    {
        using Result = std::vector<ResultValue>;

        Result result;

        fern::nullary_local_operation<Algorithm>(
            InputNoDataPolicy(), output_no_data_policy,
            fern::parallel, result);

        BOOST_CHECK(result.empty());
    }
}


BOOST_AUTO_TEST_CASE(array_1d_masked)
{
    using Result = fern::MaskedArray<ResultValue, 1>;
    using InputNoDataPolicy = fern::DetectNoDataByValue<fern::Mask<1>>;
    using OutputNoDataPolicy = fern::MarkNoDataByValue<fern::Mask<1>>;

    {
        Result result(3);

        InputNoDataPolicy input_no_data_policy(result.mask(), true);
        OutputNoDataPolicy output_no_data_policy(result.mask(), true);

        result.fill(3);

        fern::nullary_local_operation<Algorithm>(
            input_no_data_policy, output_no_data_policy,
            fern::sequential, result);

        BOOST_REQUIRE_EQUAL(result.mask()[0], false);
        BOOST_REQUIRE_EQUAL(result.mask()[1], false);
        BOOST_REQUIRE_EQUAL(result.mask()[2], false);
        BOOST_REQUIRE_EQUAL(result[0], 1);
        BOOST_REQUIRE_EQUAL(result[1], 1);
        BOOST_REQUIRE_EQUAL(result[2], 1);

        result.fill(3);
        result.mask()[1] = true;

        fern::nullary_local_operation<Algorithm>(
            input_no_data_policy, output_no_data_policy,
            fern::sequential, result);

        BOOST_REQUIRE_EQUAL(result.mask()[0], false);
        BOOST_REQUIRE_EQUAL(result.mask()[1], true);
        BOOST_REQUIRE_EQUAL(result.mask()[2], false);
        BOOST_REQUIRE_EQUAL(result[0], 1);
        BOOST_REQUIRE_EQUAL(result[1], 3);
        BOOST_REQUIRE_EQUAL(result[2], 1);
    }

    // empty
    {
        Result result;

        InputNoDataPolicy input_no_data_policy(result.mask(), true);
        OutputNoDataPolicy output_no_data_policy(result.mask(), true);

        fern::nullary_local_operation<Algorithm>(
            input_no_data_policy, output_no_data_policy,
            fern::sequential, result);

        BOOST_CHECK(result.empty());
    }
}


BOOST_AUTO_TEST_CASE(array_2d_sequential)
{
    using InputNoDataPolicy = fern::SkipNoData;
    using OutputNoDataPolicy = fern::DontMarkNoData;
    using Result = fern::Array<ResultValue, 2>;

    OutputNoDataPolicy output_no_data_policy;

    Result result{
        { 3, 3 },
        { 3, 3 },
        { 3, 3 }
    };

    fern::nullary_local_operation<Algorithm>(
        InputNoDataPolicy(), output_no_data_policy,
        fern::sequential, result);

    BOOST_CHECK_EQUAL(result[0][0], 1);
    BOOST_CHECK_EQUAL(result[0][1], 1);
    BOOST_CHECK_EQUAL(result[1][0], 1);
    BOOST_CHECK_EQUAL(result[1][1], 1);
    BOOST_CHECK_EQUAL(result[2][0], 1);
    BOOST_CHECK_EQUAL(result[2][1], 1);
}


BOOST_AUTO_TEST_CASE(array_2d_parallel)
{
    using InputNoDataPolicy = fern::SkipNoData;
    using OutputNoDataPolicy = fern::DontMarkNoData;
    using Result = fern::Array<ResultValue, 2>;

    OutputNoDataPolicy output_no_data_policy;

    fern::ThreadClient client(2);

    Result result{
        { 3, 3 },
        { 3, 3 },
        { 3, 3 }
    };

    fern::nullary_local_operation<Algorithm>(
        InputNoDataPolicy(), output_no_data_policy,
        fern::parallel, result);

    BOOST_CHECK_EQUAL(result[0][0], 1);
    BOOST_CHECK_EQUAL(result[0][1], 1);
    BOOST_CHECK_EQUAL(result[1][0], 1);
    BOOST_CHECK_EQUAL(result[1][1], 1);
    BOOST_CHECK_EQUAL(result[2][0], 1);
    BOOST_CHECK_EQUAL(result[2][1], 1);
}


BOOST_AUTO_TEST_CASE(array_2d_masked)
{
    using Result = fern::MaskedArray<ResultValue, 2>;
    using InputNoDataPolicy = fern::DetectNoDataByValue<fern::Mask<2>>;
    using OutputNoDataPolicy = fern::MarkNoDataByValue<fern::Mask<2>>;

    {
        Result result{
            { 3, 3 },
            { 3, 3 },
            { 3, 3 }
        };

        InputNoDataPolicy input_no_data_policy(result.mask(), true);
        OutputNoDataPolicy output_no_data_policy(result.mask(), true);

        result.fill(3);

        fern::nullary_local_operation<Algorithm>(
            input_no_data_policy, output_no_data_policy,
            fern::sequential, result);

        BOOST_REQUIRE_EQUAL(result.mask()[0][0], false);
        BOOST_REQUIRE_EQUAL(result.mask()[0][0], false);
        BOOST_REQUIRE_EQUAL(result.mask()[1][0], false);
        BOOST_REQUIRE_EQUAL(result.mask()[1][1], false);
        BOOST_REQUIRE_EQUAL(result.mask()[2][0], false);
        BOOST_REQUIRE_EQUAL(result.mask()[2][1], false);
        BOOST_CHECK_EQUAL(result[0][0], 1);
        BOOST_CHECK_EQUAL(result[0][1], 1);
        BOOST_CHECK_EQUAL(result[1][0], 1);
        BOOST_CHECK_EQUAL(result[1][1], 1);
        BOOST_CHECK_EQUAL(result[2][0], 1);
        BOOST_CHECK_EQUAL(result[2][1], 1);

        result.fill(3);
        result.mask()[1][1] = true;

        fern::nullary_local_operation<Algorithm>(
            input_no_data_policy, output_no_data_policy,
            fern::sequential, result);

        BOOST_REQUIRE_EQUAL(result.mask()[0][0], false);
        BOOST_REQUIRE_EQUAL(result.mask()[0][0], false);
        BOOST_REQUIRE_EQUAL(result.mask()[1][0], false);
        BOOST_REQUIRE_EQUAL(result.mask()[1][1], true);
        BOOST_REQUIRE_EQUAL(result.mask()[2][0], false);
        BOOST_REQUIRE_EQUAL(result.mask()[2][1], false);
        BOOST_CHECK_EQUAL(result[0][0], 1);
        BOOST_CHECK_EQUAL(result[0][1], 1);
        BOOST_CHECK_EQUAL(result[1][0], 1);
        BOOST_CHECK_EQUAL(result[1][1], 3);
        BOOST_CHECK_EQUAL(result[2][0], 1);
        BOOST_CHECK_EQUAL(result[2][1], 1);
    }

    // empty
    {
        Result result;

        InputNoDataPolicy input_no_data_policy(result.mask(), true);
        OutputNoDataPolicy output_no_data_policy(result.mask(), true);

        fern::nullary_local_operation<Algorithm>(
            input_no_data_policy, output_no_data_policy,
            fern::sequential, result);

        BOOST_CHECK(result.empty());
    }
}

BOOST_AUTO_TEST_SUITE_END()
