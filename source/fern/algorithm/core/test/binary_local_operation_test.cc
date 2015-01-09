#define BOOST_TEST_MODULE fern algorithm algebra binary_local_operation
#include <boost/test/unit_test.hpp>
#include "fern/core/constant_traits.h"
#include "fern/core/vector_traits.h"
#include "fern/feature/core/array_traits.h"
#include "fern/feature/core/masked_array_traits.h"
#include "fern/feature/core/masked_constant_traits.h"
#include "fern/algorithm/algebra/elementary/add.h"


namespace fa = fern::algorithm;

using ArgumentValue = int32_t;
using ResultValue = int32_t;

template<
    class Value1,
    class Value2
>
using Algorithm = fa::add::detail::Algorithm<Value1, Value2>;

template<
    class Value1,
    class Value2,
    class Result
>
using OutOfRangePolicy = fa::add::OutOfRangePolicy<Value1, Value2, Result>;


BOOST_AUTO_TEST_SUITE(binary_local_operation)

BOOST_AUTO_TEST_CASE(d0_array_d0_array)
{
    using InputNoDataPolicy = fa::InputNoDataPolicies<fa::SkipNoData<>,
          fa::SkipNoData<>>;
    using OutputNoDataPolicy = fa::DontMarkNoData;
    using Argument1 = ArgumentValue;
    using Argument2 = ArgumentValue;
    using Result = ResultValue;

    OutputNoDataPolicy output_no_data_policy;
    Argument1 argument1{-5};
    Argument2 argument2{-6};
    Result result{3};

    fa::binary_local_operation<
        Algorithm,
        fa::binary::DiscardDomainErrors,
        fa::binary::DiscardRangeErrors>(
            InputNoDataPolicy{{}, {}},
            output_no_data_policy,
            fa::sequential, argument1, argument2, result);

    BOOST_CHECK_EQUAL(result, -11);

    result = 3;

    fa::binary_local_operation<
        Algorithm,
        fa::binary::DiscardDomainErrors,
        fa::binary::DiscardRangeErrors>(
            InputNoDataPolicy{{}, {}},
            output_no_data_policy,
            fa::parallel, argument1, argument2, result);

    BOOST_CHECK_EQUAL(result, -11);
}


BOOST_AUTO_TEST_CASE(masked_d0_array_d0_array)
{
    using InputNoDataPolicy = fa::InputNoDataPolicies<
        fa::DetectNoDataByValue<bool>, fa::DetectNoDataByValue<bool>>;
    using OutputNoDataPolicy = fa::MarkNoDataByValue<bool>;
    using Argument1 = fern::MaskedConstant<ArgumentValue>;
    using Argument2 = fern::MaskedConstant<ArgumentValue>;
    using Result = fern::MaskedConstant<ResultValue>;

    Argument1 argument1;
    Argument2 argument2;
    Result result;

    // Input is not masked.
    {
        argument1.value() = -5;
        argument1.mask() = false;
        argument2.value() = -6;
        argument2.mask() = false;
        result.value() = 3;
        result.mask() = false;

        InputNoDataPolicy input_no_data_policy{
            {argument1.mask(), true},
            {argument2.mask(), true}};
        OutputNoDataPolicy output_no_data_policy(result.mask(), true);

        fa::binary_local_operation<
            Algorithm,
            fa::binary::DiscardDomainErrors,
            fa::binary::DiscardRangeErrors>(
                input_no_data_policy,
                output_no_data_policy,
                fa::sequential, argument1, argument2, result);

        BOOST_CHECK_EQUAL(result.mask(), false);
        BOOST_CHECK_EQUAL(result.value(), -11);
    }

    // Input is masked.
    {
        argument1.value() = -5;
        argument1.mask() = true;
        argument2.value() = -6;
        argument2.mask() = false;
        result.value() = 3;
        result.mask() = false;

        InputNoDataPolicy input_no_data_policy{
            {argument1.mask(), true},
            {argument2.mask(), true}};
        OutputNoDataPolicy output_no_data_policy(result.mask(), true);

        fa::binary_local_operation<
            Algorithm,
            fa::binary::DiscardDomainErrors,
            fa::binary::DiscardRangeErrors>(
                input_no_data_policy,
                output_no_data_policy,
                fa::sequential, argument1, argument2, result);

        BOOST_CHECK_EQUAL(result.mask(), true);
        BOOST_CHECK_EQUAL(result.value(), 3);
    }

    // Input is out of domain.
    {
        // TODO  Use other operation.
    }

    // Result goes out of range.
    {
        argument1.value() = fern::max<ArgumentValue>();
        argument1.mask() = false;
        argument2.value() = 1;
        argument2.mask() = false;
        result.value() = 3;
        result.mask() = false;

        InputNoDataPolicy input_no_data_policy{
            {argument1.mask(), true},
            {argument2.mask(), true}};
        OutputNoDataPolicy output_no_data_policy(result.mask(), true);
        fa::binary_local_operation<
            Algorithm,
            fa::binary::DiscardDomainErrors,
            OutOfRangePolicy>(
                input_no_data_policy,
                output_no_data_policy,
                fa::sequential, argument1, argument2, result);

        BOOST_CHECK_EQUAL(result.mask(), true);

        // Result value is max<ArgumentValue> + 1, which equals
        // min<ArgumentValue>.
        BOOST_CHECK_EQUAL(result.value(), fern::min<ArgumentValue>());
    }
}


BOOST_AUTO_TEST_CASE(d1_array_d0_array_sequential)
{
    using InputNoDataPolicy = fa::InputNoDataPolicies<fa::SkipNoData<>,
          fa::SkipNoData<>>;
    using OutputNoDataPolicy = fa::DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;

    // vector
    {
        using Argument1 = std::vector<ArgumentValue>;
        using Argument2 = ArgumentValue;
        using Result = std::vector<ResultValue>;

        Argument1 argument1{-5, 0, 5};
        Argument2 argument2{6};
        Result result{3, 3, 3};

        fa::binary_local_operation<
            Algorithm,
            fa::binary::DiscardDomainErrors,
            fa::binary::DiscardRangeErrors>(
                InputNoDataPolicy{{}, {}},
                output_no_data_policy,
                fa::sequential, argument1, argument2, result);

        BOOST_CHECK_EQUAL(result[0], 1);
        BOOST_CHECK_EQUAL(result[1], 6);
        BOOST_CHECK_EQUAL(result[2], 11);

        std::fill(result.begin(), result.end(), 3);

        // Switch arguments.
        fa::binary_local_operation<
            Algorithm,
            fa::binary::DiscardDomainErrors,
            fa::binary::DiscardRangeErrors>(
                InputNoDataPolicy{{}, {}},
                output_no_data_policy,
                fa::sequential, argument2, argument1, result);

        BOOST_CHECK_EQUAL(result[0], 1);
        BOOST_CHECK_EQUAL(result[1], 6);
        BOOST_CHECK_EQUAL(result[2], 11);
    }

    // 1d array
    {
        using Argument1 = fern::Array<ArgumentValue, 1>;
        using Argument2 = ArgumentValue;
        using Result = fern::Array<ResultValue, 1>;

        Argument1 argument1{-5, 0, 5};
        Argument2 argument2{6};
        Result result{3, 3, 3};

        fa::binary_local_operation<
            Algorithm,
            fa::binary::DiscardDomainErrors,
            fa::binary::DiscardRangeErrors>(
                InputNoDataPolicy{{}, {}},
                output_no_data_policy,
                fa::sequential, argument1, argument2, result);

        BOOST_CHECK_EQUAL(result[0], 1);
        BOOST_CHECK_EQUAL(result[1], 6);
        BOOST_CHECK_EQUAL(result[2], 11);
    }

    // empty
    {
        using Argument1 = std::vector<ArgumentValue>;
        using Argument2 = ArgumentValue;
        using Result = std::vector<ResultValue>;

        Argument1 argument1;
        Argument2 argument2{};
        Result result;

        fa::binary_local_operation<
            Algorithm,
            fa::binary::DiscardDomainErrors,
            fa::binary::DiscardRangeErrors>(
                InputNoDataPolicy{{}, {}},
                output_no_data_policy,
                fa::sequential, argument1, argument2, result);

        BOOST_CHECK(result.empty());
    }
}


/// BOOST_AUTO_TEST_CASE(array_1d_parallel)
/// {
///     using InputNoDataPolicy = fern::SkipNoData<>;
///     using OutputNoDataPolicy = fern::DontMarkNoData;
/// 
///     OutputNoDataPolicy output_no_data_policy;
/// 
///     fern::ThreadClient client(2);
/// 
///     // vector
///     {
///         using Argument = std::vector<ArgumentValue>;
///         using Result = std::vector<ResultValue>;
/// 
///         Argument argument{-5, 0, 5};
///         Result result{3, 3, 3};
/// 
///         fa::unary_local_operation<
///             Algorithm,
///             fern::unary::DiscardDomainErrors,
///             fern::unary::DiscardRangeErrors>(
///                 InputNoDataPolicy(),
///                 output_no_data_policy,
///                 fa::parallel, argument, result);
/// 
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
/// 
///         fa::unary_local_operation<
///             Algorithm,
///             fern::unary::DiscardDomainErrors,
///             fern::unary::DiscardRangeErrors>(
///                 InputNoDataPolicy(),
///                 output_no_data_policy,
///                 fa::parallel, argument, result);
/// 
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
/// 
///         fa::unary_local_operation<
///             Algorithm,
///             fern::unary::DiscardDomainErrors,
///             fern::unary::DiscardRangeErrors>(
///                 InputNoDataPolicy(),
///                 output_no_data_policy,
///                 fa::parallel, argument, result);
/// 
///         BOOST_CHECK(result.empty());
///     }
/// }
/// 
/// 
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
/// 
///         InputNoDataPolicy input_no_data_policy(result.mask(), true);
///         OutputNoDataPolicy output_no_data_policy(result.mask(), true);
/// 
///         result.fill(3);
/// 
///         fern::unary_local_operation<
///             Algorithm,
///             fern::unary::DiscardDomainErrors,
///             fern::unary::DiscardRangeErrors>(
///                 input_no_data_policy,
///                 output_no_data_policy,
///                 fern::sequential, argument, result);
/// 
///         BOOST_REQUIRE_EQUAL(result.mask()[0], false);
///         BOOST_REQUIRE_EQUAL(result.mask()[1], false);
///         BOOST_REQUIRE_EQUAL(result.mask()[2], false);
///         BOOST_REQUIRE_EQUAL(result[0], 5);
///         BOOST_REQUIRE_EQUAL(result[1], 0);
///         BOOST_REQUIRE_EQUAL(result[2], 5);
/// 
///         result.fill(3);
///         result.mask()[1] = true;
/// 
///         fern::unary_local_operation<
///             Algorithm,
///             fern::unary::DiscardDomainErrors,
///             fern::unary::DiscardRangeErrors>(
///                 input_no_data_policy,
///                 output_no_data_policy,
///                 fern::sequential, argument, result);
/// 
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
/// 
///         InputNoDataPolicy input_no_data_policy(result.mask(), true);
///         OutputNoDataPolicy output_no_data_policy(result.mask(), true);
/// 
///         fern::unary_local_operation<
///             Algorithm,
///             fern::unary::DiscardDomainErrors,
///             fern::unary::DiscardRangeErrors>(
///                 input_no_data_policy,
///                 output_no_data_policy,
///                 fern::sequential, argument, result);
/// 
///         BOOST_CHECK(result.empty());
///     }
/// }
/// 
/// 
/// BOOST_AUTO_TEST_CASE(array_2d_sequential)
/// {
///     using InputNoDataPolicy = fern::SkipNoData<>;
///     using OutputNoDataPolicy = fern::DontMarkNoData;
///     using Argument = fern::Array<ArgumentValue, 2>;
///     using Result = fern::Array<ResultValue, 2>;
/// 
///     OutputNoDataPolicy output_no_data_policy;
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
/// 
///     fern::unary_local_operation<
///         Algorithm,
///         fern::unary::DiscardDomainErrors,
///         fern::unary::DiscardRangeErrors>(
///             InputNoDataPolicy(),
///             output_no_data_policy,
///             fern::sequential, argument, result);
/// 
///     BOOST_CHECK_EQUAL(result[0][0], 2);
///     BOOST_CHECK_EQUAL(result[0][1], 1);
///     BOOST_CHECK_EQUAL(result[1][0], 0);
///     BOOST_CHECK_EQUAL(result[1][1], 9);
///     BOOST_CHECK_EQUAL(result[2][0], 1);
///     BOOST_CHECK_EQUAL(result[2][1], 2);
/// }
/// 
/// 
/// BOOST_AUTO_TEST_CASE(array_2d_parallel)
/// {
///     using InputNoDataPolicy = fern::SkipNoData<>;
///     using OutputNoDataPolicy = fern::DontMarkNoData;
///     using Argument = fern::Array<ArgumentValue, 2>;
///     using Result = fern::Array<ResultValue, 2>;
/// 
///     OutputNoDataPolicy output_no_data_policy;
/// 
///     fern::ThreadClient client(2);
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
/// 
///     fern::unary_local_operation<
///         Algorithm,
///         fern::unary::DiscardDomainErrors,
///         fern::unary::DiscardRangeErrors>(
///             InputNoDataPolicy(),
///             output_no_data_policy,
///             fern::parallel, argument, result);
/// 
///     BOOST_CHECK_EQUAL(result[0][0], 2);
///     BOOST_CHECK_EQUAL(result[0][1], 1);
///     BOOST_CHECK_EQUAL(result[1][0], 0);
///     BOOST_CHECK_EQUAL(result[1][1], 9);
///     BOOST_CHECK_EQUAL(result[2][0], 1);
///     BOOST_CHECK_EQUAL(result[2][1], 2);
/// }
/// 
/// 
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
/// 
///         InputNoDataPolicy input_no_data_policy(result.mask(), true);
///         OutputNoDataPolicy output_no_data_policy(result.mask(), true);
/// 
///         result.fill(3);
/// 
///         fern::unary_local_operation<
///             Algorithm,
///             fern::unary::DiscardDomainErrors,
///             fern::unary::DiscardRangeErrors>(
///                 input_no_data_policy,
///                 output_no_data_policy,
///                 fern::sequential, argument, result);
/// 
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
/// 
///         fern::unary_local_operation<
///             Algorithm,
///             fern::unary::DiscardDomainErrors,
///             fern::unary::DiscardRangeErrors>(
///                 input_no_data_policy,
///                 output_no_data_policy,
///                 fern::sequential, argument, result);
/// 
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
/// 
///         InputNoDataPolicy input_no_data_policy(result.mask(), true);
///         OutputNoDataPolicy output_no_data_policy(result.mask(), true);
/// 
///         fern::unary_local_operation<
///             Algorithm,
///             fern::unary::DiscardDomainErrors,
///             fern::unary::DiscardRangeErrors>(
///                 input_no_data_policy,
///                 output_no_data_policy,
///                 fern::sequential, argument, result);
/// 
///         BOOST_CHECK(result.empty());
///     }
/// }

BOOST_AUTO_TEST_SUITE_END()



/// #include "fern/feature/core/array_traits.h"
/// #include "fern/feature/core/masked_array_traits.h"
/// #include "fern/feature/core/masked_constant_traits.h"
/// #include "fern/feature/core/test/masked_constant.h"
/// #include "fern/core/thread_client.h"
/// #include "fern/core/vector_traits.h"
/// #include "fern/algorithm/policy/policies.h"
/// #include "fern/algorithm/statistic/count.h"
/// #include "fern/algorithm/algebra/executor.h"
/// #include "fern/algorithm/algebra/elementary/equal.h"



// BOOST_AUTO_TEST_CASE(masked_d0_array_d0_array)
// {
//     fern::MaskedConstant<int8_t> constant1;
//     fern::MaskedConstant<int32_t> constant2;
//     fern::MaskedConstant<int32_t> result;
// 
//     // MaskedConstants with non-masking equal. ---------------------------------
//     // Constants are not masked.
//     constant1.mask() = false;
//     constant1.value() = 5;
//     constant2.mask() = false;
//     constant2.value() = 5;
//     result.value() = 9;
//     result.mask() = false;
//     fern::algebra::add(constant1, constant2, result);
//     BOOST_CHECK(!result.mask());
//     BOOST_CHECK_EQUAL(result.value(), 10);
// 
//     // Constant is masked.
//     constant1.mask() = true;
//     constant1.value() = 5;
//     constant2.mask() = false;
//     constant2.value() = 5;
//     result.value() = 9;
//     result.mask() = false;
//     fern::algebra::add(constant1, constant2, result);
//     BOOST_CHECK(!result.mask());
//     BOOST_CHECK_EQUAL(result.value(), 10);
// 
//     // MaskedConstant with masking add. ----------------------------------------
//     using InputNoDataPolicy = fern::DetectNoDataByValue<bool>;
//     using OutputNoDataPolicy = fern::MarkNoDataByValue<bool>;
// 
//     // Constants are not masked.
//     constant1.mask() = false;
//     constant1.value() = 5;
//     constant2.mask() = false;
//     constant2.value() = 5;
//     result.value() = 9;
//     result.mask() = false;
//     fern::algebra::add<fern::MaskedConstant<int8_t>,
//         fern::MaskedConstant<int32_t>, fern::MaskedConstant<int32_t>,
//         fern::binary::DiscardDomainErrors, fern::binary::DiscardRangeErrors, InputNoDataPolicy,
//         OutputNoDataPolicy>(
//             InputNoDataPolicy(result.mask(), true),
//             OutputNoDataPolicy(result.mask(), true),
//             constant1, constant2, result);
//     BOOST_CHECK(!result.mask());
//     BOOST_CHECK_EQUAL(result.value(), 10);
// 
//     // Constants are masked.
//     constant1.value() = 5;
//     constant1.mask() = true;
//     constant2.value() = 5;
//     constant2.mask() = false;
//     result.value() = 9;
//     result.mask() = constant1.mask() || constant2.mask();
//     fern::algebra::add<fern::MaskedConstant<int8_t>,
//         fern::MaskedConstant<int32_t>, fern::MaskedConstant<int32_t>,
//         fern::binary::DiscardDomainErrors, fern::binary::DiscardRangeErrors, InputNoDataPolicy,
//         OutputNoDataPolicy>(
//             InputNoDataPolicy(result.mask(), true),
//             OutputNoDataPolicy(result.mask(), true),
//             constant1, constant2, result);
//     BOOST_CHECK(result.mask());
//     BOOST_CHECK_EQUAL(result.value(), 9);
// }
// 
// 
// BOOST_AUTO_TEST_CASE(d1_array_d1_array)
// {
//     // vector
//     {
//         std::vector<int32_t> array1{1, 2, 3};
//         std::vector<uint64_t> array2{1, 4, 3};
//         std::vector<int64_t> result(3);
//         fern::algebra::add(array1, array2, result);
//         BOOST_CHECK_EQUAL(result[0], 2);
//         BOOST_CHECK_EQUAL(result[1], 6);
//         BOOST_CHECK_EQUAL(result[2], 6);
//     }
// 
//     // 1d array
//     {
//         fern::Array<uint8_t, 1> array1{1, 2, 3};
//         fern::Array<uint16_t, 1> array2{1, 4, 3};
//         std::vector<uint16_t> result(3);
//         fern::algebra::add(array1, array2, result);
//         BOOST_CHECK_EQUAL(result[0], 2);
//         BOOST_CHECK_EQUAL(result[1], 6);
//         BOOST_CHECK_EQUAL(result[2], 6);
//     }
// 
//     // empty
//     {
//         std::vector<int32_t> array1;
//         std::vector<int32_t> array2;
//         std::vector<int32_t> result;
//         fern::algebra::add(array1, array2, result);
//         BOOST_CHECK(result.empty());
//     }
// }
// 
// 
// BOOST_AUTO_TEST_CASE(masked_d1_array_masked_d1_array)
// {
//     using InputNoDataPolicy = fern::DetectNoDataByValue<fern::Mask<1>>;
//     using OutputNoDataPolicy = fern::MarkNoDataByValue<fern::Mask<1>>;
// 
//     fern::MaskedArray<int32_t, 1> array1{1, 2, 3};
//     fern::MaskedArray<int32_t, 1> array2{1, 4, 3};
// 
//     // 1d masked arrays with non-masking add
//     {
//         fern::MaskedArray<int32_t, 1> result(3);
//         fern::algebra::add(array1, array2, result);
//         BOOST_CHECK_EQUAL(result.mask()[0], false);
//         BOOST_CHECK_EQUAL(result.mask()[1], false);
//         BOOST_CHECK_EQUAL(result.mask()[2], false);
//         BOOST_CHECK_EQUAL(result[0], 2);
//         BOOST_CHECK_EQUAL(result[1], 6);
//         BOOST_CHECK_EQUAL(result[2], 6);
//     }
// 
//     // 1d masked arrays with masking add
//     {
//         fern::MaskedArray<int32_t, 1> result(3);
//         result.mask()[2] = true;
//         fern::algebra::add<
//             fern::MaskedArray<int32_t, 1>,
//             fern::MaskedArray<int32_t, 1>,
//             fern::MaskedArray<int32_t, 1>,
//             fern::binary::DiscardDomainErrors,
//             fern::binary::DiscardRangeErrors,
//             InputNoDataPolicy,
//             OutputNoDataPolicy>(
//                 InputNoDataPolicy(result.mask(), true),
//                 OutputNoDataPolicy(result.mask(), true),
//                 array1, array2, result);
//         BOOST_CHECK_EQUAL(result.mask()[0], false);
//         BOOST_CHECK_EQUAL(result.mask()[1], false);
//         BOOST_CHECK_EQUAL(result.mask()[2], true);
//         BOOST_CHECK_EQUAL(result[0], 2);
//         BOOST_CHECK_EQUAL(result[1], 6);
//         BOOST_CHECK_EQUAL(result[2], 0);  // <-- Not touched.
// 
//         // Mask the whole input. Result must be masked too.
//         result.mask_all();
//         fern::algebra::add<
//             fern::MaskedArray<int32_t, 1>,
//             fern::MaskedArray<int32_t, 1>,
//             fern::MaskedArray<int32_t, 1>,
//             fern::binary::DiscardDomainErrors,
//             fern::binary::DiscardRangeErrors,
//             InputNoDataPolicy,
//             OutputNoDataPolicy>(
//                 InputNoDataPolicy(result.mask(), true),
//                 OutputNoDataPolicy(result.mask(), true),
//                 array1, array2, result);
//         BOOST_CHECK_EQUAL(result.mask()[0], true);
//         BOOST_CHECK_EQUAL(result.mask()[1], true);
//         BOOST_CHECK_EQUAL(result.mask()[2], true);
//     }
// 
//     // empty
//     {
//         fern::MaskedArray<int32_t, 1> empty_array;
//         fern::MaskedArray<int32_t, 1> result;
//         fern::algebra::add<
//             fern::MaskedArray<int32_t, 1>,
//             fern::MaskedArray<int32_t, 1>,
//             fern::MaskedArray<int32_t, 1>,
//             fern::binary::DiscardDomainErrors,
//             fern::binary::DiscardRangeErrors,
//             InputNoDataPolicy,
//             OutputNoDataPolicy>(
//                 InputNoDataPolicy(result.mask(), true),
//                 OutputNoDataPolicy(result.mask(), true),
//                 empty_array, empty_array, result);
//         BOOST_CHECK(result.empty());
//     }
// }
// 
// 
// BOOST_AUTO_TEST_CASE(d2_array_d2_array)
// {
//     // 2d array
//     {
//         fern::Array<int8_t, 2> array1{
//             { -2, -1 },
//             {  0,  9 },
//             {  1,  2 }
//         };
//         fern::Array<int8_t, 2> array2{
//             { -2, -1 },
//             {  5,  9 },
//             {  1,  2 }
//         };
//         fern::Array<int8_t, 2> result{
//             { 0, 0 },
//             { 0, 0 },
//             { 0, 0 }
//         };
//         fern::algebra::add(array1, array2, result);
//         BOOST_CHECK_EQUAL(result[0][0], -4);
//         BOOST_CHECK_EQUAL(result[0][1], -2);
//         BOOST_CHECK_EQUAL(result[1][0], 5);
//         BOOST_CHECK_EQUAL(result[1][1], 18);
//         BOOST_CHECK_EQUAL(result[2][0], 2);
//         BOOST_CHECK_EQUAL(result[2][1], 4);
//     }
// }
// 
// 
// BOOST_AUTO_TEST_CASE(masked_d2_array_masked_d2_array)
// {
//     fern::MaskedArray<int8_t, 2> array1{
//         { -2, -1 },
//         {  0,  9 },
//         {  1,  2 }
//     };
//     fern::MaskedArray<int8_t, 2> array2{
//         { -2, -1 },
//         {  5,  9 },
//         {  1,  2 }
//     };
// 
//     // 2d masked array with non-masking add
//     {
//         fern::MaskedArray<int8_t, 2> result{
//             { 0, 0 },
//             { 0, 0 },
//             { 0, 0 }
//         };
//         fern::algebra::add(array1, array2, result);
//         BOOST_CHECK_EQUAL(result[0][0], -4);
//         BOOST_CHECK_EQUAL(result[0][1], -2);
//         BOOST_CHECK_EQUAL(result[1][0], 5);
//         BOOST_CHECK_EQUAL(result[1][1], 18);
//         BOOST_CHECK_EQUAL(result[2][0], 2);
//         BOOST_CHECK_EQUAL(result[2][1], 4);
//     }
// 
//     // 2d masked arrays with masking add
//     {
//         // TODO
//     }
// }
// 
// 
// BOOST_AUTO_TEST_CASE(concurrent_d2_array)
// {
//     size_t const nr_rows = 6000;
//     size_t const nr_cols = 4000;
//     auto const extents = fern::extents[nr_rows][nr_cols];
//     fern::Array<int32_t, 2> array1(extents);
//     fern::Array<int32_t, 2> array2(extents);
//     fern::Array<int32_t, 2> plus_result_we_get(extents);
//     fern::Array<int32_t, 2> plus_result_we_want(extents);
//     fern::Array<bool, 2> equal_result_we_get(extents, true);
//     size_t count_result_we_get;
//     size_t const count_result_we_want = nr_rows * nr_cols;
//     fern::algebra::Add<
//         fern::Array<int32_t, 2>,
//         fern::Array<int32_t, 2>,
//         fern::Array<int32_t, 2>> add;
//     fern::algebra::Equal<
//         fern::Array<int32_t, 2>,
//         fern::Array<int32_t, 2>,
//         fern::Array<bool, 2>> equal;
//     fern::statistic::Count<
//         fern::Array<bool, 2>,
//         size_t> count;
// 
//     std::iota(array1.data(), array1.data() + array1.num_elements(), 0);
//     std::iota(array2.data(), array2.data() + array2.num_elements(), 0);
//     std::transform(array1.data(), array1.data() + array1.num_elements(),
//         plus_result_we_want.data(), [](int32_t value){ return value + value; });
// 
//     // Serial.
//     {
//         fern::serial::execute(add, array1, array2, plus_result_we_get);
//         fern::serial::execute(equal, plus_result_we_get, plus_result_we_want,
//             equal_result_we_get);
//         fern::serial::execute(count, equal_result_we_get, true,
//             count_result_we_get);
//         BOOST_CHECK_EQUAL(count_result_we_get, count_result_we_want);
//     }
// 
//     // Concurrent.
//     {
//         fern::ThreadClient client;
//         fern::concurrent::execute(add, array1, array2, plus_result_we_get);
//         fern::concurrent::execute(equal, plus_result_we_get,
//             plus_result_we_want, equal_result_we_get);
//         fern::concurrent::execute(count, equal_result_we_get, true,
//             count_result_we_get);
//         BOOST_CHECK_EQUAL(count_result_we_get, count_result_we_want);
//     }
// }
// 
// 
// BOOST_AUTO_TEST_CASE(concurrent_masked_d2_array)
// {
//     using InputNoDataPolicy = fern::SkipNoData<>;
//     using OutputNoDataPolicy = fern::MarkNoDataByValue<fern::Mask<2>>;
// 
//     size_t const nr_rows = 6000;
//     size_t const nr_cols = 4000;
//     auto const extents = fern::extents[nr_rows][nr_cols];
//     fern::MaskedArray<int32_t, 2> array1(extents);
//     fern::MaskedArray<int32_t, 2> array2(extents);
//     fern::MaskedArray<int32_t, 2> plus_result_we_get(extents);
//     fern::MaskedArray<int32_t, 2> plus_result_we_want(extents);
//     fern::MaskedArray<bool, 2> equal_result_we_get(extents, true);
//     fern::MaskedConstant<size_t> count_result_we_get;
//     fern::MaskedConstant<size_t> const count_result_we_want{nr_rows * nr_cols};
//     fern::algebra::Add<
//         fern::MaskedArray<int32_t, 2>,
//         fern::MaskedArray<int32_t, 2>,
//         fern::MaskedArray<int32_t, 2>,
//         fern::binary::DiscardDomainErrors,
//         fern::binary::DiscardRangeErrors,
//         InputNoDataPolicy,
//         OutputNoDataPolicy> add(
//             InputNoDataPolicy(),
//             OutputNoDataPolicy(plus_result_we_get.mask(), true));
//     fern::algebra::Equal<
//         fern::MaskedArray<int32_t, 2>,
//         fern::MaskedArray<int32_t, 2>,
//         fern::MaskedArray<bool, 2>,
//         fern::binary::DiscardDomainErrors,
//         fern::binary::DiscardRangeErrors,
//         InputNoDataPolicy,
//         OutputNoDataPolicy> equal(
//             InputNoDataPolicy(),
//             OutputNoDataPolicy(equal_result_we_get.mask(), true));
//     fern::statistic::Count<
//         fern::MaskedArray<bool, 2>,
//         fern::MaskedConstant<size_t>,
//         InputNoDataPolicy,
//         fern::MarkNoDataByValue<bool>> count(
//             InputNoDataPolicy(),
//             fern::MarkNoDataByValue<bool>(count_result_we_get.mask(), true));
// 
//     std::iota(array1.data(), array1.data() + array1.num_elements(), 0);
//     std::iota(array2.data(), array2.data() + array2.num_elements(), 0);
//     std::transform(array1.data(), array1.data() + array1.num_elements(),
//         plus_result_we_want.data(), [](int32_t value){ return value + value; });
// 
//     // Verify executor can handle masked result.
//     fern::ThreadClient client;
//     fern::concurrent::execute(add, array1, array2, plus_result_we_get);
//     fern::concurrent::execute(equal, plus_result_we_get,
//         plus_result_we_want, equal_result_we_get);
//     fern::concurrent::execute(count, equal_result_we_get, true,
//         count_result_we_get);
//     BOOST_CHECK_EQUAL(count_result_we_get, count_result_we_want);
// }
// 
// 
// // TODO Test 1d array + 0d array
// //      Test 2d array + 1d array
// //      etc
// // TODO Stuff below.









/// BOOST_AUTO_TEST_CASE(value)
/// {
///     verify_value<int8_t, int8_t, int8_t>(-5, 6, 1);
/// 
///     verify_value<uint16_t, int8_t>(fern::TypeTraits<uint16_t>::max, 2,
///         int32_t(fern::TypeTraits<uint16_t>::max) + int32_t(2));
/// 
///     verify_value<uint32_t, int8_t>(fern::TypeTraits<uint32_t>::max, 2,
///         int64_t(fern::TypeTraits<uint32_t>::max) + int64_t(2));
/// 
///     verify_value<uint64_t, int64_t>(
///         fern::TypeTraits<uint64_t>::max,
///         fern::TypeTraits<int64_t>::max,
///         int64_t(fern::TypeTraits<uint64_t>::max) +
///             fern::TypeTraits<int64_t>::max);
/// }
/// 
/// 
/// template<
///     class A1,
///     class A2>
/// struct DomainPolicyHost:
///     public fern::add::OutOfDomainPolicy<A1, A2>
/// {
/// };
/// 
/// 
/// BOOST_AUTO_TEST_CASE(domain)
/// {
///     {
///         DomainPolicyHost<int32_t, int32_t> domain;
///         BOOST_CHECK(domain.within_domain(-1, 2));
///     }
///     {
///         DomainPolicyHost<uint8_t, double> domain;
///         BOOST_CHECK(domain.within_domain(1, 2.0));
///     }
/// }
/// 
/// 
/// template<
///     class A1,
///     class A2>
/// struct RangePolicyHost:
///     public fern::add::OutOfRangePolicy<A1, A2>
/// {
/// };
/// 
/// 
/// template<
///     class A1,
///     class A2>
/// void verify_range_check(
///     A1 const& argument1,
///     A2 const& argument2,
///     bool const within)
/// {
///     fern::algebra::Add<A1, A2> operation;
///     typename fern::algebra::Add<A1, A2>::R result;
///     RangePolicyHost<A1, A2> range;
/// 
///     operation(argument1, argument2, result);
///     BOOST_CHECK_EQUAL((range.within_range(argument1, argument2, result)),
///         within);
/// }
/// 
/// 
/// BOOST_AUTO_TEST_CASE(range)
/// {
///     int8_t const min_int8 = fern::TypeTraits<int8_t>::min;
///     int8_t const max_int8 = fern::TypeTraits<int8_t>::max;
///     uint8_t const max_uint8 = fern::TypeTraits<uint8_t>::max;
///     uint16_t const max_uint16 = fern::TypeTraits<uint16_t>::max;
///     uint32_t const max_uint32 = fern::TypeTraits<uint32_t>::max;
///     int64_t const min_int64 = fern::TypeTraits<int64_t>::min;
///     int64_t const max_int64 = fern::TypeTraits<int64_t>::max;
///     uint64_t const max_uint64 = fern::TypeTraits<uint64_t>::max;
/// 
///     // signed + signed
///     verify_range_check<int8_t, int8_t>(-5, 6, true);
///     verify_range_check<int8_t, int8_t>(max_int8, 1, false);
///     verify_range_check<int8_t, int8_t>(min_int8, -1, false);
///     verify_range_check<int64_t, int64_t>(min_int64, -1, false);
/// 
///     // unsigned + unsigned
///     verify_range_check<uint8_t, uint8_t>(5, 6, true);
///     verify_range_check<uint8_t, uint8_t>(max_uint8, 1, false);
///     verify_range_check<uint8_t, uint16_t>(max_uint8, 1, true);
/// 
///     // signed + unsigned
///     // unsigned + signed
///     verify_range_check<int8_t, uint8_t>(5, 6, true);
///     verify_range_check<uint8_t, int8_t>(5, 6, true);
///     verify_range_check<uint16_t, int8_t>(max_uint16, 2, true);
///     verify_range_check<uint32_t, int8_t>(max_uint32, 2, true);
///     verify_range_check<uint64_t, int64_t>(max_uint64, max_int64, false);
/// 
///     // float + float
///     float const max_float32 = fern::TypeTraits<float>::max;
///     verify_range_check<float, float>(5.0, 6.0, true);
///     verify_range_check<float, float>(max_float32, max_float32 * 20, false);
/// 
///     // float + signed
///     // unsigned + float
///     verify_range_check<float, int8_t>(5.0, 6, true);
///     verify_range_check<uint8_t, float>(5, 6.0, true);
/// }
/// 
/// 
/// BOOST_AUTO_TEST_CASE(argument_types)
/// {
///     // Verify that we can pass in all kinds of collection types.
/// 
///     // constant + constant
///     {
///         uint8_t argument1(5);
///         uint8_t argument2(6);
///         using R = fern::Result<uint8_t, uint8_t>::type;
///         R result;
/// 
///         fern::algebra::add(argument1, argument2, result);
/// 
///         BOOST_CHECK_EQUAL(result, 11u);
///     }
/// 
///     // constant + vector
///     {
///         uint8_t argument1(5);
///         std::vector<uint8_t> argument2({1, 2, 3});
///         using R = fern::Result<uint8_t, uint8_t>::type;
///         std::vector<R> result(argument2.size());
/// 
///         fern::algebra::add(argument1, argument2, result);
/// 
///         BOOST_REQUIRE_EQUAL(result.size(), 3u);
///         BOOST_CHECK_EQUAL(result[0], 6u);
///         BOOST_CHECK_EQUAL(result[1], 7u);
///         BOOST_CHECK_EQUAL(result[2], 8u);
///     }
/// 
///     // vector + constant
///     {
///         std::vector<uint8_t> argument1({1, 2, 3});
///         uint8_t argument2(5);
///         using R = fern::Result<uint8_t, uint8_t>::type;
///         std::vector<R> result(argument1.size());
/// 
///         fern::algebra::add(argument1, argument2, result);
/// 
///         BOOST_REQUIRE_EQUAL(result.size(), 3u);
///         BOOST_CHECK_EQUAL(result[0], 6u);
///         BOOST_CHECK_EQUAL(result[1], 7u);
///         BOOST_CHECK_EQUAL(result[2], 8u);
///     }
/// 
///     // vector + vector
///     {
///         std::vector<uint8_t> argument1({1, 2, 3});
///         std::vector<uint8_t> argument2({4, 5, 6});
///         using R = fern::Result<uint8_t, uint8_t>::type;
///         std::vector<R> result(argument1.size());
/// 
///         fern::algebra::add(argument1, argument2, result);
/// 
///         BOOST_REQUIRE_EQUAL(result.size(), 3u);
///         BOOST_CHECK_EQUAL(result[0], 5u);
///         BOOST_CHECK_EQUAL(result[1], 7u);
///         BOOST_CHECK_EQUAL(result[2], 9u);
///     }
/// 
///     // array + array
///     {
///         fern::Array<int8_t, 2> argument(fern::extents[3][2]);
///         argument[0][0] = -2;
///         argument[0][1] = -1;
///         argument[1][0] =  0;
///         argument[1][1] =  9;
///         argument[2][0] =  1;
///         argument[2][1] =  2;
///         using R = fern::Result<int8_t, int8_t>::type;
///         fern::Array<R, 2> result(fern::extents[3][2]);
/// 
///         fern::algebra::add(argument, argument, result);
/// 
///         BOOST_CHECK_EQUAL(result[0][0], -4);
///         BOOST_CHECK_EQUAL(result[0][1], -2);
///         BOOST_CHECK_EQUAL(result[1][0],  0);
///         BOOST_CHECK_EQUAL(result[1][1], 18);
///         BOOST_CHECK_EQUAL(result[2][0],  2);
///         BOOST_CHECK_EQUAL(result[2][1],  4);
///     }
/// 
///     // masked_array + masked_array
///     {
///         fern::MaskedArray<int8_t, 2> argument(fern::extents[3][2]);
///         argument[0][0] = -2;
///         argument[0][1] = -1;
///         argument[1][0] =  0;
///         argument.mask()[1][1] =  true;
///         argument[1][1] =  9;
///         argument[2][0] =  1;
///         argument[2][1] =  2;
///         using R = fern::Result<int8_t, int8_t>::type;
///         fern::MaskedArray<R, 2> result(fern::extents[3][2]);
/// 
///         fern::algebra::add(argument, argument, result);
/// 
///         BOOST_CHECK(!result.mask()[0][0]);
///         BOOST_CHECK_EQUAL(result[0][0], -4);
/// 
///         BOOST_CHECK(!result.mask()[0][1]);
///         BOOST_CHECK_EQUAL(result[0][1], -2);
/// 
///         BOOST_CHECK(!result.mask()[1][0]);
///         BOOST_CHECK_EQUAL(result[1][0],  0);
/// 
///         // Although the input data has a mask, the default policy discards
///         // it. So the result doesn't have masked values.
///         BOOST_CHECK(!result.mask()[1][1]);
///         BOOST_CHECK_EQUAL(result[1][1], 18);
/// 
///         BOOST_CHECK(!result.mask()[2][0]);
///         BOOST_CHECK_EQUAL(result[2][0],  2);
///         BOOST_CHECK(!result.mask()[2][1]);
///         BOOST_CHECK_EQUAL(result[2][1],  4);
///     }
/// }
/// 
/// 
/// BOOST_AUTO_TEST_CASE(no_data)
/// {
///     size_t const nr_rows = 3;
///     size_t const nr_cols = 2;
///     auto extents = fern::extents[nr_rows][nr_cols];
/// 
///     fern::MaskedArray<int8_t, 2> argument1(extents);
///     argument1[0][0] = -2;
///     argument1[0][1] = -1;
///     argument1[1][0] =  0;
///     argument1.mask()[1][1] =  true;
///     argument1[2][0] =  1;
///     argument1[2][1] =  2;
/// 
///     fern::MaskedArray<int8_t, 2> argument2(extents);
///     argument2[0][0] = -2;
///     argument2[0][1] = -1;
///     argument2[1][0] =  0;
///     argument2[1][1] =  9;
///     argument2.mask()[2][0] =  true;
///     argument2[2][1] =  2;
/// 
///     int8_t argument3 = 5;
/// 
///     // masked_array + masked_array
///     {
///         // Create room for the result.
///         // Set the mask.
///         using R = fern::Result<int8_t, int8_t>::type;
///         fern::MaskedArray<R, 2> result(extents);
///         result.set_mask(argument1.mask(), true);
///         result.set_mask(argument2.mask(), true);
/// 
///         using A1 = decltype(argument1);
///         using A2 = decltype(argument2);
///         using A1Value = fern::ArgumentTraits<A1>::value_type;
///         using A2Value = fern::ArgumentTraits<A2>::value_type;
///         using OutOfDomainPolicy = fern::DiscardDomainErrors<A1Value, A2Value>;
///         using OutOfRangePolicy = fern::add::OutOfRangePolicy<A1Value, A2Value>;
///         using NoDataPolicy = fern::MarkNoDataByValue<bool, fern::Mask<2>>;
///         using Add = fern::algebra::Add<A1, A2, OutOfDomainPolicy, OutOfRangePolicy,
///             NoDataPolicy>;
/// 
///         Add add(NoDataPolicy(result.mask(), true));
/// 
///         add(argument1, argument2, result);
/// 
///         BOOST_CHECK(!result.mask()[0][0]);
///         BOOST_CHECK_EQUAL(result[0][0], -4);
/// 
///         BOOST_CHECK(!result.mask()[0][1]);
///         BOOST_CHECK_EQUAL(result[0][1], -2);
/// 
///         BOOST_CHECK(!result.mask()[1][0]);
///         BOOST_CHECK_EQUAL(result[1][0],  0);
/// 
///         BOOST_CHECK( result.mask()[1][1]);
///         // Value is masked: it is undefined.
///         // BOOST_CHECK_EQUAL(result[1][1], 18);
/// 
///         BOOST_CHECK(result.mask()[2][0]);
///         // Value is masked.
///         // BOOST_CHECK_EQUAL(result[2][0],  2);
/// 
///         BOOST_CHECK(!result.mask()[2][1]);
///         BOOST_CHECK_EQUAL(result[2][1],  4);
///     }
/// 
///     // masked_array + 5
///     {
///         // Create room for the result.
///         // Set the mask.
///         using R = fern::Result<int8_t, int8_t>::type;
///         fern::MaskedArray<R, 2> result(extents);
///         result.set_mask(argument1.mask(), true);
/// 
///         using A1 = decltype(argument1);
///         using A2 = decltype(argument3);
///         using A1Value = fern::ArgumentTraits<A1>::value_type;
///         using A2Value = fern::ArgumentTraits<A2>::value_type;
///         using OutOfDomainPolicy = fern::DiscardDomainErrors<A1Value, A2Value>;
///         using OutOfRangePolicy = fern::add::OutOfRangePolicy<A1Value, A2Value>;
///         using NoDataPolicy = fern::MarkNoDataByValue<bool, fern::Mask<2>>;
///         using Add = fern::algebra::Add<A1, A2, OutOfDomainPolicy, OutOfRangePolicy,
///             NoDataPolicy>;
/// 
///         Add add(NoDataPolicy(result.mask(), true));
/// 
///         add(argument1, argument3, result);
/// 
///         BOOST_CHECK(!result.mask()[0][0]);
///         BOOST_CHECK_EQUAL(result[0][0], 3);
/// 
///         BOOST_CHECK(!result.mask()[0][1]);
///         BOOST_CHECK_EQUAL(result[0][1], 4);
/// 
///         BOOST_CHECK(!result.mask()[1][0]);
///         BOOST_CHECK_EQUAL(result[1][0], 5);
/// 
///         BOOST_CHECK( result.mask()[1][1]);
/// 
///         BOOST_CHECK(!result.mask()[2][0]);
///         BOOST_CHECK_EQUAL(result[2][0], 6);
/// 
///         BOOST_CHECK(!result.mask()[2][1]);
///         BOOST_CHECK_EQUAL(result[2][1], 7);
///     }
/// 
///     // 5 + masked_array
///     {
///         // Create room for the result.
///         // Set the mask.
///         using R = fern::Result<int8_t, int8_t>::type;
///         fern::MaskedArray<R, 2> result(extents);
///         result.set_mask(argument1.mask(), true);
/// 
///         using A1 = decltype(argument3);
///         using A2 = decltype(argument1);
///         using A1Value = fern::ArgumentTraits<A1>::value_type;
///         using A2Value = fern::ArgumentTraits<A2>::value_type;
///         using OutOfDomainPolicy = fern::DiscardDomainErrors<A1Value, A2Value>;
///         using OutOfRangePolicy = fern::add::OutOfRangePolicy<A1Value, A2Value>;
///         using NoDataPolicy = fern::MarkNoDataByValue<bool, fern::Mask<2>>;
///         using Add = fern::algebra::Add<A1, A2, OutOfDomainPolicy, OutOfRangePolicy,
///             NoDataPolicy>;
/// 
///         Add add(NoDataPolicy(result.mask(), true));
/// 
///         add(argument3, argument1, result);
/// 
///         BOOST_CHECK(!result.mask()[0][0]);
///         BOOST_CHECK_EQUAL(result[0][0], 3);
/// 
///         BOOST_CHECK(!result.mask()[0][1]);
///         BOOST_CHECK_EQUAL(result[0][1], 4);
/// 
///         BOOST_CHECK(!result.mask()[1][0]);
///         BOOST_CHECK_EQUAL(result[1][0], 5);
/// 
///         BOOST_CHECK( result.mask()[1][1]);
/// 
///         BOOST_CHECK(!result.mask()[2][0]);
///         BOOST_CHECK_EQUAL(result[2][0], 6);
/// 
///         BOOST_CHECK(!result.mask()[2][1]);
///         BOOST_CHECK_EQUAL(result[2][1], 7);
///     }
/// }
/// 
/// 
/// BOOST_AUTO_TEST_CASE(threading)
/// {
///     // Create a somewhat larger array.
///     size_t const nr_rows = 6000;
///     size_t const nr_cols = 4000;
///     auto extents = fern::extents[nr_rows][nr_cols];
///     fern::Array<int8_t, 2> argument1(extents);
/// 
///     // Fill it with 0, 1, 2, 3, ...
///     std::iota(argument1.data(), argument1.data() + argument1.num_elements(), 0);
/// 
///     int8_t argument2 = 5;
/// 
///     // Create array with values that should be in the result.
///     fern::Array<int8_t, 2> result_we_want(extents);
///     std::iota(result_we_want.data(), result_we_want.data() +
///         result_we_want.num_elements(), 5);
/// 
///     {
///         // Add collection and constant.
///         fern::algebra::Add<fern::Array<int8_t, 2>, int8_t> add;
///         fern::Array<int8_t, 2> plus_result(extents);
/// 
///         // Compare result with result we want.
///         fern::algebra::Add<fern::Array<int8_t, 2>, fern::Array<int8_t, 2>>
///             add;
///         fern::Array<bool, 2> equal_result(extents);
/// 
///         // Count the number of equal values.
///         fern::statistic::Count<fern::Array<bool, 2>, bool> count;
///         size_t count_result;
/// 
///         // Serial.
///         {
///             fern::serial::execute(add, argument1, argument2, plus_result);
///             fern::serial::execute(equal, plus_result, result_we_want,
///                 equal_result);
///             fern::serial::execute(count, equal_result, true, count_result);
/// 
///             // Make sure the number of equal values equals the number of
///             // elements.
///             BOOST_CHECK_EQUAL(count_result, equal_result.num_elements());
///         }
/// 
///         // Make sure the concurrent call have a pool to work with.
///         fern::ThreadClient client;
/// 
///         // Concurrent.
///         {
///             fern::concurrent::execute(add, argument1, argument2, plus_result);
///             fern::concurrent::execute(equal, plus_result, result_we_want,
///                 equal_result);
///             fern::concurrent::execute(count, equal_result, true, count_result);
/// 
///             // Make sure the number of equal values equals the number of
///             // elements.
///             BOOST_CHECK_EQUAL(count_result, equal_result.num_elements());
///         }
///     }
/// 
///     // TODO Make this work for masked arrays too.
/// }



/// BOOST_AUTO_TEST_CASE(masked_d0_array_d0_array)
/// {
///     fern::MaskedConstant<int8_t> constant1;
///     fern::MaskedConstant<int32_t> constant2;
///     fern::MaskedConstant<bool> result_we_get;
/// 
///     // MaskedConstants with non-masking equal. ---------------------------------
///     // Constants are not masked.
///     constant1.mask() = false;
///     constant1.value() = 5;
///     constant2.mask() = false;
///     constant2.value() = 5;
///     result_we_get.value() = false;
///     result_we_get.mask() = false;
///     fern::algebra::equal(constant1, constant2, result_we_get);
///     BOOST_CHECK(!result_we_get.mask());
///     BOOST_CHECK_EQUAL(result_we_get.value(), true);
/// 
///     // Constant is masked.
///     constant1.mask() = true;
///     constant1.value() = 5;
///     constant2.mask() = false;
///     constant2.value() = 5;
///     result_we_get.value() = false;
///     result_we_get.mask() = false;
///     fern::algebra::equal(constant1, constant2, result_we_get);
///     BOOST_CHECK(!result_we_get.mask());
///     BOOST_CHECK_EQUAL(result_we_get.value(), true);
/// 
///     // MaskedConstant with masking equal. --------------------------------------
///     using InputNoDataPolicy = fern::DetectNoDataByValue<bool>;
///     using OutputNoDataPolicy = fern::MarkNoDataByValue<bool>;
/// 
///     // Constants are not masked.
///     constant1.mask() = false;
///     constant1.value() = 5;
///     constant2.mask() = false;
///     constant2.value() = 5;
///     result_we_get.value() = false;
///     result_we_get.mask() = constant1.mask() || constant2.mask();
///     fern::algebra::equal<fern::MaskedConstant<int8_t>,
///         fern::MaskedConstant<int32_t>, fern::MaskedConstant<bool>,
///         fern::binary::DiscardDomainErrors, fern::binary::DiscardRangeErrors,
///         InputNoDataPolicy,
///         OutputNoDataPolicy>(
///             InputNoDataPolicy(result_we_get.mask(), true),
///             OutputNoDataPolicy(result_we_get.mask(), true),
///             constant1, constant2, result_we_get);
///     BOOST_CHECK(!result_we_get.mask());
///     BOOST_CHECK_EQUAL(result_we_get.value(), true);
/// 
///     // Constants are masked.
///     constant1.value() = 5;
///     constant1.mask() = true;
///     constant2.value() = 5;
///     constant2.mask() = false;
///     result_we_get.value() = false;
///     result_we_get.mask() = constant1.mask() || constant2.mask();
///     fern::algebra::equal<fern::MaskedConstant<int8_t>,
///         fern::MaskedConstant<int32_t>, fern::MaskedConstant<bool>,
///         fern::binary::DiscardDomainErrors, fern::binary::DiscardRangeErrors,
///         InputNoDataPolicy,
///         OutputNoDataPolicy>(
///             InputNoDataPolicy(result_we_get.mask(), true),
///             OutputNoDataPolicy(result_we_get.mask(), true),
///             constant1, constant2, result_we_get);
///     BOOST_CHECK(result_we_get.mask());
///     BOOST_CHECK_EQUAL(result_we_get.value(), false);
/// }
/// 
/// 
/// BOOST_AUTO_TEST_CASE(d1_array_d1_array)
/// {
///     // vector
///     {
///         std::vector<int32_t> array1{1, 2, 3};
///         std::vector<uint64_t> array2{1, 4, 3};
///         std::vector<bool> result(3);
///         fern::algebra::equal(array1, array2, result);
///         BOOST_CHECK_EQUAL(result[0], true);
///         BOOST_CHECK_EQUAL(result[1], false);
///         BOOST_CHECK_EQUAL(result[2], true);
///     }
/// 
///     // 1d array
///     {
///         fern::Array<uint8_t, 1> array1{1, 2, 3};
///         fern::Array<uint16_t, 1> array2{1, 4, 3};
///         std::vector<bool> result(3);
///         fern::algebra::equal(array1, array2, result);
///         BOOST_CHECK_EQUAL(result[0], true);
///         BOOST_CHECK_EQUAL(result[1], false);
///         BOOST_CHECK_EQUAL(result[2], true);
///     }
/// 
///     // empty
///     {
///         std::vector<int32_t> array1;
///         std::vector<int32_t> array2;
///         std::vector<bool> result;
///         fern::algebra::equal(array1, array2, result);
///         BOOST_CHECK(result.empty());
///     }
/// }
/// 
/// 
/// BOOST_AUTO_TEST_CASE(masked_d1_array)
/// {
///     using InputNoDataPolicy = fern::DetectNoDataByValue<fern::Mask<1>>;
///     using OutputNoDataPolicy = fern::MarkNoDataByValue<fern::Mask<1>>;
/// 
///     fern::MaskedArray<int32_t, 1> array1{1, 2, 3};
///     fern::MaskedArray<int32_t, 1> array2{1, 4, 3};
/// 
///     // 1d masked arrays with non-masking equal
///     {
///         fern::MaskedArray<bool, 1> result(3, false);
///         fern::algebra::equal(array1, array2, result);
///         BOOST_CHECK_EQUAL(result.mask()[0], false);
///         BOOST_CHECK_EQUAL(result.mask()[1], false);
///         BOOST_CHECK_EQUAL(result.mask()[2], false);
///         BOOST_CHECK_EQUAL(result[0], true);
///         BOOST_CHECK_EQUAL(result[1], false);
///         BOOST_CHECK_EQUAL(result[2], true);
///     }
/// 
///     // 1d masked arrays with masking equal
///     {
///         fern::MaskedArray<bool, 1> result(3, false);
///         result.mask()[2] = true;
///         fern::algebra::equal<
///             fern::MaskedArray<int32_t, 1>,
///             fern::MaskedArray<int32_t, 1>,
///             fern::MaskedArray<bool, 1>,
///             fern::binary::DiscardDomainErrors,
///             fern::binary::DiscardRangeErrors,
///             InputNoDataPolicy,
///             OutputNoDataPolicy>(
///                 InputNoDataPolicy(result.mask(), true),
///                 OutputNoDataPolicy(result.mask(), true),
///                 array1, array2, result);
///         BOOST_CHECK_EQUAL(result.mask()[0], false);
///         BOOST_CHECK_EQUAL(result.mask()[1], false);
///         BOOST_CHECK_EQUAL(result.mask()[2], true);
///         BOOST_CHECK_EQUAL(result[0], true);
///         BOOST_CHECK_EQUAL(result[1], false);
///         BOOST_CHECK_EQUAL(result[2], false);  // <-- Not touched.
/// 
///         // Mask the whole input. Result must be masked too.
///         result.mask_all();
///         fern::algebra::equal<
///             fern::MaskedArray<int32_t, 1>,
///             fern::MaskedArray<int32_t, 1>,
///             fern::MaskedArray<bool, 1>,
///             fern::binary::DiscardDomainErrors,
///             fern::binary::DiscardRangeErrors,
///             InputNoDataPolicy,
///             OutputNoDataPolicy>(
///                 InputNoDataPolicy(result.mask(), true),
///                 OutputNoDataPolicy(result.mask(), true),
///                 array1, array2, result);
///         BOOST_CHECK_EQUAL(result.mask()[0], true);
///         BOOST_CHECK_EQUAL(result.mask()[1], true);
///         BOOST_CHECK_EQUAL(result.mask()[2], true);
///     }
/// 
///     // empty
///     {
///         fern::MaskedArray<int32_t, 1> empty_array;
///         fern::MaskedArray<bool, 1> result;
///         fern::algebra::equal<
///             fern::MaskedArray<int32_t, 1>,
///             fern::MaskedArray<int32_t, 1>,
///             fern::MaskedArray<bool, 1>,
///             fern::binary::DiscardDomainErrors,
///             fern::binary::DiscardRangeErrors,
///             InputNoDataPolicy,
///             OutputNoDataPolicy>(
///                 InputNoDataPolicy(result.mask(), true),
///                 OutputNoDataPolicy(result.mask(), true),
///                 empty_array, empty_array, result);
///         BOOST_CHECK(result.empty());
///     }
/// }
/// 
/// 
/// BOOST_AUTO_TEST_CASE(d2_array_d2_array)
/// {
///     // 2d array
///     {
///         fern::Array<int8_t, 2> array1{
///             { -2, -1 },
///             {  0,  9 },
///             {  1,  2 }
///         };
///         fern::Array<int8_t, 2> array2{
///             { -2, -1 },
///             {  5,  9 },
///             {  1,  2 }
///         };
///         fern::Array<bool, 2> result{
///             { false, false },
///             { false, false },
///             { false, false }
///         };
///         fern::algebra::equal(array1, array2, result);
///         BOOST_CHECK_EQUAL(result[0][0], true);
///         BOOST_CHECK_EQUAL(result[0][1], true);
///         BOOST_CHECK_EQUAL(result[1][0], false);
///         BOOST_CHECK_EQUAL(result[1][1], true);
///         BOOST_CHECK_EQUAL(result[2][0], true);
///         BOOST_CHECK_EQUAL(result[2][1], true);
///     }
/// }
/// 
/// 
/// BOOST_AUTO_TEST_CASE(masked_d2_array_d2_array)
/// {
///     fern::MaskedArray<int8_t, 2> array1{
///         { -2, -1 },
///         {  0,  9 },
///         {  1,  2 }
///     };
///     fern::MaskedArray<int8_t, 2> array2{
///         { -2, -1 },
///         {  5,  9 },
///         {  1,  2 }
///     };
/// 
///     // 2d masked array with non-masking equal
///     {
///         fern::MaskedArray<bool, 2> result{
///             { true, true },
///             { true, true },
///             { true, true }
///         };
///         fern::algebra::equal(array1, array2, result);
///         BOOST_CHECK_EQUAL(result[0][0], true);
///         BOOST_CHECK_EQUAL(result[0][1], true);
///         BOOST_CHECK_EQUAL(result[1][0], false);
///         BOOST_CHECK_EQUAL(result[1][1], true);
///         BOOST_CHECK_EQUAL(result[2][0], true);
///         BOOST_CHECK_EQUAL(result[2][1], true);
///     }
/// 
///     // 2d masked arrays with masking equal
///     {
///         // TODO
///     }
/// }
/// 
/// 
/// BOOST_AUTO_TEST_CASE(concurrent)
/// {
///     // Create a somewhat larger array.
///     size_t const nr_rows = 6000;
///     size_t const nr_cols = 4000;
///     auto const extents = fern::extents[nr_rows][nr_cols];
///     fern::Array<int32_t, 2> array1(extents);
///     fern::Array<int32_t, 2> array2(extents);
///     fern::Array<bool, 2> result(extents);
///     fern::algebra::Equal<
///         fern::Array<int32_t, 2>,
///         fern::Array<int32_t, 2>,
///         fern::Array<bool, 2>> equal;
/// 
///     std::iota(array1.data(), array1.data() + array1.num_elements(), 0);
///     std::iota(array2.data(), array2.data() + array2.num_elements(), 0);
/// 
///     // Serial.
///     {
///         result.fill(false);
///         fern::serial::execute(equal, array1, array2, result);
///         BOOST_CHECK(std::all_of(result.data(), result.data() +
///             result.num_elements(), [](bool equal){ return equal; }));
///     }
/// 
///     // Concurrent.
///     {
///         result.fill(false);
///         fern::ThreadClient client;
///         fern::concurrent::execute(equal, array1, array2, result);
///         BOOST_CHECK(std::all_of(result.data(), result.data() +
///             result.num_elements(), [](bool equal){ return equal; }));
///     }
/// 
///     {
///         using InputNoDataPolicy = fern::SkipNoData<>;
///         using OutputNoDataPolicy = fern::MarkNoDataByValue<fern::Mask<2>>;
/// 
///         fern::MaskedArray<bool, 2> result(extents);
///         fern::algebra::Equal<
///             fern::Array<int32_t, 2>,
///             fern::Array<int32_t, 2>,
///             fern::MaskedArray<bool, 2>,
///             fern::binary::DiscardDomainErrors,
///             fern::binary::DiscardRangeErrors,
///             InputNoDataPolicy,
///             OutputNoDataPolicy> equal(
///                 InputNoDataPolicy(),
///                 OutputNoDataPolicy(result.mask(), true));
/// 
///         // Verify executor can handle masked result.
///         fern::ThreadClient client;
///         fern::concurrent::execute(equal, array1, array2, result);
///         BOOST_CHECK(std::all_of(result.data(), result.data() +
///             result.num_elements(), [](bool equal){ return equal; }));
///     }
/// }
/// 
/// // TODO Test 1d array == 0d array
/// //      Test 2d array == 1d array
/// //      etc
