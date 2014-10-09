#define BOOST_TEST_MODULE fern algorithm algebra nullary_local_operation
#include <boost/test/unit_test.hpp>
#include "fern/core/constant_traits.h"
#include "fern/core/vector_traits.h"
#include "fern/feature/core/array_traits.h"
#include "fern/feature/core/masked_array_traits.h"
#include "fern/feature/core/masked_constant_traits.h"
#include "fern/algorithm/algebra/boole/defined.h"


namespace fa = fern::algorithm;


struct Fixture
{

    Fixture()
        : _thread_client(2)
    {
    }

    fern::ThreadClient _thread_client;

};


using ArgumentValue = int32_t;
using ResultValue = int32_t;
using Algorithm = fa::defined::detail::Algorithm;


BOOST_FIXTURE_TEST_SUITE(nullary_local_operation, Fixture)

BOOST_AUTO_TEST_CASE(array_0d)
{
    using InputNoDataPolicy = fa::SkipNoData<>;
    using OutputNoDataPolicy = fa::DontMarkNoData;
    using Result = ResultValue;

    OutputNoDataPolicy output_no_data_policy;
    Result result{3};

    fa::nullary_local_operation<Algorithm>(
        InputNoDataPolicy(), output_no_data_policy,
        fa::sequential, result);

    BOOST_REQUIRE_EQUAL(result, 1);

    result = 3;

    fa::nullary_local_operation<Algorithm>(
        InputNoDataPolicy(), output_no_data_policy,
        fa::parallel, result);

    BOOST_REQUIRE_EQUAL(result, 1);
}


BOOST_AUTO_TEST_CASE(array_0d_masked)
{
    using InputNoDataPolicy = fa::DetectNoDataByValue<bool>;
    using OutputNoDataPolicy = fa::MarkNoDataByValue<bool>;
    using Result = fern::MaskedConstant<ResultValue>;

    Result result;

    // Input is not masked.
    {
        result.value() = 3;
        result.mask() = false;

        InputNoDataPolicy input_no_data_policy(result.mask(), true);
        OutputNoDataPolicy output_no_data_policy(result.mask(), true);

        fa::nullary_local_operation<Algorithm>(
            input_no_data_policy, output_no_data_policy,
            fa::sequential, result);

        BOOST_REQUIRE_EQUAL(result.mask(), false);
        BOOST_REQUIRE_EQUAL(result.value(), 1);
    }

    // Input is masked.
    {
        result.value() = 3;
        result.mask() = true;

        InputNoDataPolicy input_no_data_policy(result.mask(), true);
        OutputNoDataPolicy output_no_data_policy(result.mask(), true);
        fa::nullary_local_operation<Algorithm>(
            input_no_data_policy, output_no_data_policy,
            fa::sequential, result);

        BOOST_REQUIRE_EQUAL(result.mask(), true);
        BOOST_REQUIRE_EQUAL(result.value(), 3);
    }
}


BOOST_AUTO_TEST_CASE(array_1d_sequential)
{
    using InputNoDataPolicy = fa::SkipNoData<>;
    using OutputNoDataPolicy = fa::DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;

    // vector
    {
        using Result = std::vector<ResultValue>;

        Result result{3, 3, 3};

        fa::nullary_local_operation<Algorithm>(
            InputNoDataPolicy(), output_no_data_policy,
            fa::sequential, result);

        BOOST_REQUIRE_EQUAL(result[0], 1);
        BOOST_REQUIRE_EQUAL(result[1], 1);
        BOOST_REQUIRE_EQUAL(result[2], 1);
    }

    // 1d array
    {
        using Result = fern::Array<ResultValue, 1>;

        Result result{3, 3, 3};

        fa::nullary_local_operation<Algorithm>(
            InputNoDataPolicy(), output_no_data_policy,
            fa::sequential, result);

        BOOST_REQUIRE_EQUAL(result[0], 1);
        BOOST_REQUIRE_EQUAL(result[1], 1);
        BOOST_REQUIRE_EQUAL(result[2], 1);
    }

    // empty
    {
        using Result = std::vector<ResultValue>;

        Result result;

        fa::nullary_local_operation<Algorithm>(
            InputNoDataPolicy(), output_no_data_policy,
            fa::sequential, result);

        BOOST_CHECK(result.empty());
    }
}


BOOST_AUTO_TEST_CASE(array_1d_parallel)
{
    using InputNoDataPolicy = fa::SkipNoData<>;
    using OutputNoDataPolicy = fa::DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;

    // vector
    {
        using Result = std::vector<ResultValue>;

        Result result{3, 3, 3};

        fa::nullary_local_operation<Algorithm>(
            InputNoDataPolicy(), output_no_data_policy,
            fa::parallel, result);

        BOOST_REQUIRE_EQUAL(result[0], 1);
        BOOST_REQUIRE_EQUAL(result[1], 1);
        BOOST_REQUIRE_EQUAL(result[2], 1);
    }

    // 1d array
    {
        using Result = fern::Array<ResultValue, 1>;

        Result result{3, 3, 3};

        fa::nullary_local_operation<Algorithm>(
            InputNoDataPolicy(), output_no_data_policy,
            fa::parallel, result);

        BOOST_REQUIRE_EQUAL(result[0], 1);
        BOOST_REQUIRE_EQUAL(result[1], 1);
        BOOST_REQUIRE_EQUAL(result[2], 1);
    }

    // empty
    {
        using Result = std::vector<ResultValue>;

        Result result;

        fa::nullary_local_operation<Algorithm>(
            InputNoDataPolicy(), output_no_data_policy,
            fa::parallel, result);

        BOOST_CHECK(result.empty());
    }
}


BOOST_AUTO_TEST_CASE(array_1d_masked)
{
    using Result = fern::MaskedArray<ResultValue, 1>;
    using InputNoDataPolicy = fa::DetectNoDataByValue<fern::Mask<1>>;
    using OutputNoDataPolicy = fa::MarkNoDataByValue<fern::Mask<1>>;

    {
        Result result(3);

        InputNoDataPolicy input_no_data_policy(result.mask(), true);
        OutputNoDataPolicy output_no_data_policy(result.mask(), true);

        result.fill(3);

        fa::nullary_local_operation<Algorithm>(
            input_no_data_policy, output_no_data_policy,
            fa::sequential, result);

        BOOST_REQUIRE_EQUAL(result.mask()[0], false);
        BOOST_REQUIRE_EQUAL(result.mask()[1], false);
        BOOST_REQUIRE_EQUAL(result.mask()[2], false);
        BOOST_REQUIRE_EQUAL(result[0], 1);
        BOOST_REQUIRE_EQUAL(result[1], 1);
        BOOST_REQUIRE_EQUAL(result[2], 1);

        result.fill(3);
        result.mask()[1] = true;

        fa::nullary_local_operation<Algorithm>(
            input_no_data_policy, output_no_data_policy,
            fa::sequential, result);

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

        fa::nullary_local_operation<Algorithm>(
            input_no_data_policy, output_no_data_policy,
            fa::sequential, result);

        BOOST_CHECK(result.empty());
    }
}


BOOST_AUTO_TEST_CASE(array_2d_sequential)
{
    using InputNoDataPolicy = fa::SkipNoData<>;
    using OutputNoDataPolicy = fa::DontMarkNoData;
    using Result = fern::Array<ResultValue, 2>;

    OutputNoDataPolicy output_no_data_policy;

    Result result{
        { 3, 3 },
        { 3, 3 },
        { 3, 3 }
    };

    fa::nullary_local_operation<Algorithm>(
        InputNoDataPolicy(), output_no_data_policy,
        fa::sequential, result);

    BOOST_CHECK_EQUAL(result[0][0], 1);
    BOOST_CHECK_EQUAL(result[0][1], 1);
    BOOST_CHECK_EQUAL(result[1][0], 1);
    BOOST_CHECK_EQUAL(result[1][1], 1);
    BOOST_CHECK_EQUAL(result[2][0], 1);
    BOOST_CHECK_EQUAL(result[2][1], 1);
}


BOOST_AUTO_TEST_CASE(array_2d_parallel)
{
    using InputNoDataPolicy = fa::SkipNoData<>;
    using OutputNoDataPolicy = fa::DontMarkNoData;
    using Result = fern::Array<ResultValue, 2>;

    OutputNoDataPolicy output_no_data_policy;

    Result result{
        { 3, 3 },
        { 3, 3 },
        { 3, 3 }
    };

    fa::nullary_local_operation<Algorithm>(
        InputNoDataPolicy(), output_no_data_policy,
        fa::parallel, result);

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
    using InputNoDataPolicy = fa::DetectNoDataByValue<fern::Mask<2>>;
    using OutputNoDataPolicy = fa::MarkNoDataByValue<fern::Mask<2>>;

    {
        Result result{
            { 3, 3 },
            { 3, 3 },
            { 3, 3 }
        };

        InputNoDataPolicy input_no_data_policy(result.mask(), true);
        OutputNoDataPolicy output_no_data_policy(result.mask(), true);

        result.fill(3);

        fa::nullary_local_operation<Algorithm>(
            input_no_data_policy, output_no_data_policy,
            fa::sequential, result);

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

        fa::nullary_local_operation<Algorithm>(
            input_no_data_policy, output_no_data_policy,
            fa::sequential, result);

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

        fa::nullary_local_operation<Algorithm>(
            input_no_data_policy, output_no_data_policy,
            fa::sequential, result);

        BOOST_CHECK(result.empty());
    }
}

BOOST_AUTO_TEST_SUITE_END()