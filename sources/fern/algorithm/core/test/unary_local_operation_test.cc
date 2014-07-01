#define BOOST_TEST_MODULE fern algorithm algebra unary_local_operation
#include <boost/test/unit_test.hpp>
#include "fern/core/constant_traits.h"
#include "fern/core/vector_traits.h"
#include "fern/feature/core/array_traits.h"
#include "fern/feature/core/masked_array_traits.h"
#include "fern/feature/core/masked_constant_traits.h"
#include "fern/algorithm/algebra/elementary/absolute.h"


using ArgumentValue = int32_t;
using ResultValue = int32_t;

template<
    class ArgumentValue
>
using Algorithm = fern::absolute::detail::Algorithm<ArgumentValue>;

template<
    class Value,
    class Result
>
using OutOfRangePolicy = fern::absolute::OutOfRangePolicy<Value, Result>;


/// template<
///     class Argument,
///     class Result,
///     class InputNoDataPolicy=fern::SkipNoData,
///     class OutputNoDataPolicy=fern::DontMarkNoData
/// >
/// using UnaryLocalOperation = fern::detail::dispatch::UnaryLocalOperation<
///     Argument, Result,
///     fern::absolute::OutOfDomainPolicy, fern::absolute::OutOfRangePolicy,
///     InputNoDataPolicy, OutputNoDataPolicy,
///     Algorithm<ArgumentValue>,
///     typename fern::ArgumentTraits<Argument>::argument_category>;


BOOST_AUTO_TEST_SUITE(unary_local_operation)

/// BOOST_AUTO_TEST_CASE(array_0d)
/// {
///     using Argument = ArgumentValue;
///     using Result = ResultValue;
/// 
///     Argument argument;
///     Result result;
///     UnaryLocalOperation<Argument, Result> operation{
///         Algorithm<ArgumentValue>()};
/// 
///     argument = -5;
///     result = 3;
///     operation.calculate(argument, result);
///     BOOST_REQUIRE_EQUAL(result, 5);
/// }


BOOST_AUTO_TEST_CASE(array_0d)
{
    using InputNoDataPolicy = fern::SkipNoData;
    using OutputNoDataPolicy = fern::DontMarkNoData;
    using Argument = ArgumentValue;
    using Result = ResultValue;

    OutputNoDataPolicy output_no_data_policy;
    Argument argument{-5};
    Result result{3};

    fern::unary_local_operation<
        Algorithm,
        fern::unary::DiscardDomainErrors,
        fern::unary::DiscardRangeErrors>(
            InputNoDataPolicy(),
            output_no_data_policy,
            fern::sequential, argument, result);

    BOOST_REQUIRE_EQUAL(result, 5);
}


/// BOOST_AUTO_TEST_CASE(array_0d_masked)
/// {
///     using Argument = fern::MaskedConstant<ArgumentValue>;
///     using Result = fern::MaskedConstant<ResultValue>;
/// 
///     Argument argument;
///     Result result;
///     UnaryLocalOperation<Argument, Result,
///         fern::DetectNoDataByValue<bool>,
///         fern::MarkNoDataByValue<bool>> operation(
///             fern::DetectNoDataByValue<bool>(result.mask(), true),
///             fern::MarkNoDataByValue<bool>(result.mask(), true),
///             Algorithm<ArgumentValue>());
/// 
///     // Input is masked.
///     {
///         argument.value() = -5;
///         argument.mask() = false;
///         result.value() = 3;
///         result.mask() = argument.mask();
///         operation.calculate(argument, result);
///         BOOST_REQUIRE_EQUAL(result.mask(), false);
///         BOOST_REQUIRE_EQUAL(result.value(), 5);
///     }
/// 
///     // Input is not masked.
///     {
///         argument.value() = -5;
///         argument.mask() = true;
///         result.value() = 3;
///         result.mask() = argument.mask();
///         operation.calculate(argument, result);
///         BOOST_REQUIRE_EQUAL(result.mask(), true);
///         BOOST_REQUIRE_EQUAL(result.value(), 3);
///     }
/// }


BOOST_AUTO_TEST_CASE(array_0d_masked)
{
    using InputNoDataPolicy = fern::DetectNoDataByValue<bool>;
    using OutputNoDataPolicy = fern::MarkNoDataByValue<bool>;
    using Argument = fern::MaskedConstant<ArgumentValue>;
    using Result = fern::MaskedConstant<ResultValue>;

    Argument argument;
    Result result;

    // Input is not masked.
    {
        argument.value() = -5;
        argument.mask() = false;
        result.value() = 3;
        result.mask() = argument.mask();

        InputNoDataPolicy input_no_data_policy(result.mask(), true);
        OutputNoDataPolicy output_no_data_policy(result.mask(), true);

        fern::unary_local_operation<
            Algorithm,
            fern::unary::DiscardDomainErrors,
            fern::unary::DiscardRangeErrors>(
                input_no_data_policy,
                output_no_data_policy,
                fern::sequential, argument, result);

        BOOST_REQUIRE_EQUAL(result.mask(), false);
        BOOST_REQUIRE_EQUAL(result.value(), 5);
    }

    // Input is masked.
    {
        argument.value() = -5;
        argument.mask() = true;
        result.value() = 3;
        result.mask() = argument.mask();

        InputNoDataPolicy input_no_data_policy(result.mask(), true);
        OutputNoDataPolicy output_no_data_policy(result.mask(), true);
        fern::unary_local_operation<
            Algorithm,
            fern::unary::DiscardDomainErrors,
            fern::unary::DiscardRangeErrors>(
                input_no_data_policy,
                output_no_data_policy,
                fern::sequential, argument, result);

        BOOST_REQUIRE_EQUAL(result.mask(), true);
        BOOST_REQUIRE_EQUAL(result.value(), 3);
    }

    // Input is out of domain.
    {
        // TODO  Use sqrt, for example.
    }

    // Result goes out of range.
    {
        argument.value() = fern::min<ArgumentValue>();
        argument.mask() = false;
        result.value() = 3;
        result.mask() = argument.mask();

        InputNoDataPolicy input_no_data_policy(result.mask(), true);
        OutputNoDataPolicy output_no_data_policy(result.mask(), true);
        fern::unary_local_operation<
            Algorithm,
            fern::unary::DiscardDomainErrors,
            OutOfRangePolicy>(
                input_no_data_policy,
                output_no_data_policy,
                fern::sequential, argument, result);

        BOOST_REQUIRE_EQUAL(result.mask(), true);

        // Result value is max<ArgumentValue> + 1, which equals
        // min<ArgumentValue>.
        BOOST_REQUIRE_EQUAL(result.value(), fern::min<ArgumentValue>());
    }
}


/// BOOST_AUTO_TEST_CASE(array_1d)
/// {
///     // vector
///     {
///         using Argument = std::vector<ArgumentValue>;
///         using Result = std::vector<ResultValue>;
/// 
///         Argument argument{-5, 0, 5};
///         Result result{3, 3, 3};
///         UnaryLocalOperation<Argument, Result> operation{
///             Algorithm<ArgumentValue>()};
///         operation.calculate(argument, result);
///         BOOST_REQUIRE_EQUAL(result[0], 5);
///         BOOST_REQUIRE_EQUAL(result[1], 0);
///         BOOST_REQUIRE_EQUAL(result[2], 5);
///     }
/// 
///     // 1d array
///     {
///         using Argument = fern::Array<ArgumentValue, 1>;
///         using Result = fern::Array<ResultValue, 1>;
/// 
///         Argument argument{-5, 0, 5};
///         Result result{3, 3, 3};
///         UnaryLocalOperation<Argument, Result> operation{
///             Algorithm<ArgumentValue>()};
///         operation.calculate(argument, result);
///         BOOST_REQUIRE_EQUAL(result[0], 5);
///         BOOST_REQUIRE_EQUAL(result[1], 0);
///         BOOST_REQUIRE_EQUAL(result[2], 5);
///     }
/// 
///     // empty
///     {
///         using Argument = std::vector<ArgumentValue>;
///         using Result = std::vector<ResultValue>;
/// 
///         Argument argument;
///         Result result;
///         UnaryLocalOperation<Argument, Result> operation{
///             Algorithm<ArgumentValue>()};
///         operation.calculate(argument, result);
///         BOOST_CHECK(result.empty());
///     }
/// }


BOOST_AUTO_TEST_CASE(array_1d)
{
    using InputNoDataPolicy = fern::SkipNoData;
    using OutputNoDataPolicy = fern::DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;

    // vector
    {
        using Argument = std::vector<ArgumentValue>;
        using Result = std::vector<ResultValue>;

        Argument argument{-5, 0, 5};
        Result result{3, 3, 3};

        fern::unary_local_operation<
            Algorithm,
            fern::unary::DiscardDomainErrors,
            fern::unary::DiscardRangeErrors>(
                InputNoDataPolicy(),
                output_no_data_policy,
                fern::sequential, argument, result);

        BOOST_REQUIRE_EQUAL(result[0], 5);
        BOOST_REQUIRE_EQUAL(result[1], 0);
        BOOST_REQUIRE_EQUAL(result[2], 5);
    }

    // 1d array
    {
        using Argument = fern::Array<ArgumentValue, 1>;
        using Result = fern::Array<ResultValue, 1>;

        Argument argument{-5, 0, 5};
        Result result{3, 3, 3};

        fern::unary_local_operation<
            Algorithm,
            fern::unary::DiscardDomainErrors,
            fern::unary::DiscardRangeErrors>(
                InputNoDataPolicy(),
                output_no_data_policy,
                fern::sequential, argument, result);

        BOOST_REQUIRE_EQUAL(result[0], 5);
        BOOST_REQUIRE_EQUAL(result[1], 0);
        BOOST_REQUIRE_EQUAL(result[2], 5);
    }

    // empty
    {
        using Argument = std::vector<ArgumentValue>;
        using Result = std::vector<ResultValue>;

        Argument argument;
        Result result;

        fern::unary_local_operation<
            Algorithm,
            fern::unary::DiscardDomainErrors,
            fern::unary::DiscardRangeErrors>(
                InputNoDataPolicy(),
                output_no_data_policy,
                fern::sequential, argument, result);

        BOOST_CHECK(result.empty());
    }
}


/// BOOST_AUTO_TEST_CASE(array_1d_masked)
/// {
///     using Argument = fern::MaskedArray<ArgumentValue, 1>;
///     using Result = fern::MaskedArray<ResultValue, 1>;
///     using InputNoDataPolicy = fern::DetectNoDataByValue<fern::Mask<1>>;
///     using OutputNoDataPolicy = fern::MarkNoDataByValue<fern::Mask<1>>;
/// 
///     {
///         Argument argument{-5, 0, 5};
///         Result result(3);
///         UnaryLocalOperation<Argument, Result,
///             InputNoDataPolicy, OutputNoDataPolicy> operation(
///                 InputNoDataPolicy(result.mask(), true),
///                 OutputNoDataPolicy(result.mask(), true),
///                 Algorithm<ArgumentValue>());
/// 
///         result.fill(3);
///         operation.calculate(argument, result);
///         BOOST_REQUIRE_EQUAL(result.mask()[0], false);
///         BOOST_REQUIRE_EQUAL(result.mask()[1], false);
///         BOOST_REQUIRE_EQUAL(result.mask()[2], false);
///         BOOST_REQUIRE_EQUAL(result[0], 5);
///         BOOST_REQUIRE_EQUAL(result[1], 0);
///         BOOST_REQUIRE_EQUAL(result[2], 5);
/// 
///         result.fill(3);
///         result.mask()[1] = true;
///         operation.calculate(argument, result);
///         BOOST_REQUIRE_EQUAL(result.mask()[0], false);
///         BOOST_REQUIRE_EQUAL(result.mask()[1], true);
///         BOOST_REQUIRE_EQUAL(result.mask()[2], false);
///         BOOST_REQUIRE_EQUAL(result[0], 5);
///         BOOST_REQUIRE_EQUAL(result[1], 3);
///         BOOST_REQUIRE_EQUAL(result[2], 5);
///     }
/// 
///     // empty
///     {
///         Argument argument;
///         Result result;
///         UnaryLocalOperation<Argument, Result,
///             InputNoDataPolicy, OutputNoDataPolicy> operation(
///                 InputNoDataPolicy(result.mask(), true),
///                 OutputNoDataPolicy(result.mask(), true),
///                 Algorithm<ArgumentValue>());
/// 
///         operation.calculate(argument, result);
///         BOOST_CHECK(result.empty());
///     }
/// }


BOOST_AUTO_TEST_CASE(array_1d_masked)
{
    using Argument = fern::MaskedArray<ArgumentValue, 1>;
    using Result = fern::MaskedArray<ResultValue, 1>;
    using InputNoDataPolicy = fern::DetectNoDataByValue<fern::Mask<1>>;
    using OutputNoDataPolicy = fern::MarkNoDataByValue<fern::Mask<1>>;

    {
        Argument argument{-5, 0, 5};
        Result result(3);

        InputNoDataPolicy input_no_data_policy(result.mask(), true);
        OutputNoDataPolicy output_no_data_policy(result.mask(), true);

        result.fill(3);

        fern::unary_local_operation<
            Algorithm,
            fern::unary::DiscardDomainErrors,
            fern::unary::DiscardRangeErrors>(
                input_no_data_policy,
                output_no_data_policy,
                fern::sequential, argument, result);

        BOOST_REQUIRE_EQUAL(result.mask()[0], false);
        BOOST_REQUIRE_EQUAL(result.mask()[1], false);
        BOOST_REQUIRE_EQUAL(result.mask()[2], false);
        BOOST_REQUIRE_EQUAL(result[0], 5);
        BOOST_REQUIRE_EQUAL(result[1], 0);
        BOOST_REQUIRE_EQUAL(result[2], 5);

        result.fill(3);
        result.mask()[1] = true;

        fern::unary_local_operation<
            Algorithm,
            fern::unary::DiscardDomainErrors,
            fern::unary::DiscardRangeErrors>(
                input_no_data_policy,
                output_no_data_policy,
                fern::sequential, argument, result);

        BOOST_REQUIRE_EQUAL(result.mask()[0], false);
        BOOST_REQUIRE_EQUAL(result.mask()[1], true);
        BOOST_REQUIRE_EQUAL(result.mask()[2], false);
        BOOST_REQUIRE_EQUAL(result[0], 5);
        BOOST_REQUIRE_EQUAL(result[1], 3);
        BOOST_REQUIRE_EQUAL(result[2], 5);
    }

    // empty
    {
        Argument argument;
        Result result;

        InputNoDataPolicy input_no_data_policy(result.mask(), true);
        OutputNoDataPolicy output_no_data_policy(result.mask(), true);

        fern::unary_local_operation<
            Algorithm,
            fern::unary::DiscardDomainErrors,
            fern::unary::DiscardRangeErrors>(
                input_no_data_policy,
                output_no_data_policy,
                fern::sequential, argument, result);

        BOOST_CHECK(result.empty());
    }
}


/// BOOST_AUTO_TEST_CASE(array_2d)
/// {
///     using Argument = fern::Array<ArgumentValue, 2>;
///     using Result = fern::Array<ResultValue, 2>;
/// 
///     Argument argument{
///         { -2, -1 },
///         {  0,  9 },
///         {  1,  2 }
///     };
///     Result result{
///         { 3, 3 },
///         { 3, 3 },
///         { 3, 3 }
///     };
///     UnaryLocalOperation<Argument, Result> operation{
///         Algorithm<ArgumentValue>()};
///     operation.calculate(argument, result);
///     BOOST_CHECK_EQUAL(result[0][0], 2);
///     BOOST_CHECK_EQUAL(result[0][1], 1);
///     BOOST_CHECK_EQUAL(result[1][0], 0);
///     BOOST_CHECK_EQUAL(result[1][1], 9);
///     BOOST_CHECK_EQUAL(result[2][0], 1);
///     BOOST_CHECK_EQUAL(result[2][1], 2);
/// }


BOOST_AUTO_TEST_CASE(array_2d)
{
    using InputNoDataPolicy = fern::SkipNoData;
    using OutputNoDataPolicy = fern::DontMarkNoData;
    using Argument = fern::Array<ArgumentValue, 2>;
    using Result = fern::Array<ResultValue, 2>;

    OutputNoDataPolicy output_no_data_policy;

    Argument argument{
        { -2, -1 },
        {  0,  9 },
        {  1,  2 }
    };
    Result result{
        { 3, 3 },
        { 3, 3 },
        { 3, 3 }
    };

    // UnaryLocalOperation<Argument, Result> operation{
    //     Algorithm<ArgumentValue>()};
    // operation.calculate(argument, result);

    fern::unary_local_operation<
        Algorithm,
        fern::unary::DiscardDomainErrors,
        fern::unary::DiscardRangeErrors>(
            InputNoDataPolicy(),
            output_no_data_policy,
            fern::sequential, argument, result);

    BOOST_CHECK_EQUAL(result[0][0], 2);
    BOOST_CHECK_EQUAL(result[0][1], 1);
    BOOST_CHECK_EQUAL(result[1][0], 0);
    BOOST_CHECK_EQUAL(result[1][1], 9);
    BOOST_CHECK_EQUAL(result[2][0], 1);
    BOOST_CHECK_EQUAL(result[2][1], 2);
}


/// BOOST_AUTO_TEST_CASE(array_2d_masked)
/// {
///     using Argument = fern::MaskedArray<ArgumentValue, 2>;
///     using Result = fern::MaskedArray<ResultValue, 2>;
///     using InputNoDataPolicy = fern::DetectNoDataByValue<fern::Mask<2>>;
///     using OutputNoDataPolicy = fern::MarkNoDataByValue<fern::Mask<2>>;
/// 
///     {
///         Argument argument{
///             { -2, -1 },
///             {  0,  9 },
///             {  1,  2 }
///         };
///         Result result{
///             { 3, 3 },
///             { 3, 3 },
///             { 3, 3 }
///         };
///         UnaryLocalOperation<Argument, Result,
///             InputNoDataPolicy, OutputNoDataPolicy> operation(
///                 InputNoDataPolicy(result.mask(), true),
///                 OutputNoDataPolicy(result.mask(), true),
///                 Algorithm<ArgumentValue>());
/// 
///         result.fill(3);
///         operation.calculate(argument, result);
///         BOOST_REQUIRE_EQUAL(result.mask()[0][0], false);
///         BOOST_REQUIRE_EQUAL(result.mask()[0][0], false);
///         BOOST_REQUIRE_EQUAL(result.mask()[1][0], false);
///         BOOST_REQUIRE_EQUAL(result.mask()[1][1], false);
///         BOOST_REQUIRE_EQUAL(result.mask()[2][0], false);
///         BOOST_REQUIRE_EQUAL(result.mask()[2][1], false);
///         BOOST_CHECK_EQUAL(result[0][0], 2);
///         BOOST_CHECK_EQUAL(result[0][1], 1);
///         BOOST_CHECK_EQUAL(result[1][0], 0);
///         BOOST_CHECK_EQUAL(result[1][1], 9);
///         BOOST_CHECK_EQUAL(result[2][0], 1);
///         BOOST_CHECK_EQUAL(result[2][1], 2);
/// 
///         result.fill(3);
///         result.mask()[1][1] = true;
///         operation.calculate(argument, result);
///         BOOST_REQUIRE_EQUAL(result.mask()[0][0], false);
///         BOOST_REQUIRE_EQUAL(result.mask()[0][0], false);
///         BOOST_REQUIRE_EQUAL(result.mask()[1][0], false);
///         BOOST_REQUIRE_EQUAL(result.mask()[1][1], true);
///         BOOST_REQUIRE_EQUAL(result.mask()[2][0], false);
///         BOOST_REQUIRE_EQUAL(result.mask()[2][1], false);
///         BOOST_CHECK_EQUAL(result[0][0], 2);
///         BOOST_CHECK_EQUAL(result[0][1], 1);
///         BOOST_CHECK_EQUAL(result[1][0], 0);
///         BOOST_CHECK_EQUAL(result[1][1], 3);
///         BOOST_CHECK_EQUAL(result[2][0], 1);
///         BOOST_CHECK_EQUAL(result[2][1], 2);
///     }
/// 
///     // empty
///     {
///         Argument argument;
///         Result result;
///         UnaryLocalOperation<Argument, Result,
///             InputNoDataPolicy, OutputNoDataPolicy> operation(
///                 InputNoDataPolicy(result.mask(), true),
///                 OutputNoDataPolicy(result.mask(), true),
///                 Algorithm<ArgumentValue>());
/// 
///         operation.calculate(argument, result);
///         BOOST_CHECK(result.empty());
///     }
/// }


BOOST_AUTO_TEST_CASE(array_2d_masked)
{
    using Argument = fern::MaskedArray<ArgumentValue, 2>;
    using Result = fern::MaskedArray<ResultValue, 2>;
    using InputNoDataPolicy = fern::DetectNoDataByValue<fern::Mask<2>>;
    using OutputNoDataPolicy = fern::MarkNoDataByValue<fern::Mask<2>>;

    {
        Argument argument{
            { -2, -1 },
            {  0,  9 },
            {  1,  2 }
        };
        Result result{
            { 3, 3 },
            { 3, 3 },
            { 3, 3 }
        };

        InputNoDataPolicy input_no_data_policy(result.mask(), true);
        OutputNoDataPolicy output_no_data_policy(result.mask(), true);

        result.fill(3);

        fern::unary_local_operation<
            Algorithm,
            fern::unary::DiscardDomainErrors,
            fern::unary::DiscardRangeErrors>(
                input_no_data_policy,
                output_no_data_policy,
                fern::sequential, argument, result);

        BOOST_REQUIRE_EQUAL(result.mask()[0][0], false);
        BOOST_REQUIRE_EQUAL(result.mask()[0][0], false);
        BOOST_REQUIRE_EQUAL(result.mask()[1][0], false);
        BOOST_REQUIRE_EQUAL(result.mask()[1][1], false);
        BOOST_REQUIRE_EQUAL(result.mask()[2][0], false);
        BOOST_REQUIRE_EQUAL(result.mask()[2][1], false);
        BOOST_CHECK_EQUAL(result[0][0], 2);
        BOOST_CHECK_EQUAL(result[0][1], 1);
        BOOST_CHECK_EQUAL(result[1][0], 0);
        BOOST_CHECK_EQUAL(result[1][1], 9);
        BOOST_CHECK_EQUAL(result[2][0], 1);
        BOOST_CHECK_EQUAL(result[2][1], 2);

        result.fill(3);
        result.mask()[1][1] = true;

        fern::unary_local_operation<
            Algorithm,
            fern::unary::DiscardDomainErrors,
            fern::unary::DiscardRangeErrors>(
                input_no_data_policy,
                output_no_data_policy,
                fern::sequential, argument, result);

        BOOST_REQUIRE_EQUAL(result.mask()[0][0], false);
        BOOST_REQUIRE_EQUAL(result.mask()[0][0], false);
        BOOST_REQUIRE_EQUAL(result.mask()[1][0], false);
        BOOST_REQUIRE_EQUAL(result.mask()[1][1], true);
        BOOST_REQUIRE_EQUAL(result.mask()[2][0], false);
        BOOST_REQUIRE_EQUAL(result.mask()[2][1], false);
        BOOST_CHECK_EQUAL(result[0][0], 2);
        BOOST_CHECK_EQUAL(result[0][1], 1);
        BOOST_CHECK_EQUAL(result[1][0], 0);
        BOOST_CHECK_EQUAL(result[1][1], 3);
        BOOST_CHECK_EQUAL(result[2][0], 1);
        BOOST_CHECK_EQUAL(result[2][1], 2);
    }

    // empty
    {
        Argument argument;
        Result result;

        InputNoDataPolicy input_no_data_policy(result.mask(), true);
        OutputNoDataPolicy output_no_data_policy(result.mask(), true);

        fern::unary_local_operation<
            Algorithm,
            fern::unary::DiscardDomainErrors,
            fern::unary::DiscardRangeErrors>(
                input_no_data_policy,
                output_no_data_policy,
                fern::sequential, argument, result);

        BOOST_CHECK(result.empty());
    }
}

// TODO Also test with parallel execution policy.

BOOST_AUTO_TEST_SUITE_END()
