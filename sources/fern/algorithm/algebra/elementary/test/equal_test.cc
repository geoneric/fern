#define BOOST_TEST_MODULE fern algorithm algebra elementary equal
#include <algorithm>
#include <boost/test/unit_test.hpp>
#include "fern/feature/core/array_traits.h"
#include "fern/core/constant_traits.h"
#include "fern/core/typename.h"
#include "fern/core/vector_traits.h"
#include "fern/feature/core/masked_array_traits.h"
#include "fern/feature/core/masked_constant_traits.h"
#include "fern/algorithm/algebra/elementary/equal.h"
#include "fern/algorithm/algebra/executor.h"
#include "fern/algorithm/policy/policies.h"


template<
    class A1,
    class A2,
    class R>
void verify_value(
    A1 const& values1,
    A2 const& values2,
    R const& result_we_want)
{
    R result_we_get;
    fern::algebra::equal(values1, values2, result_we_get);
    BOOST_CHECK_EQUAL(result_we_get, result_we_want);
}


BOOST_AUTO_TEST_SUITE(equal)

BOOST_AUTO_TEST_CASE(traits)
{
    using Equal = fern::algebra::Equal<int32_t, int32_t, bool>;
    BOOST_CHECK((std::is_same<fern::OperationTraits<Equal>::category,
        fern::local_operation_tag>::value));
}


BOOST_AUTO_TEST_CASE(d0_array_d0_array)
{
    verify_value<int8_t, int8_t>(-5, 6, false);
    verify_value<int8_t, int8_t>(-5, -5, true);
    verify_value<double, double>(-5.5, -5.5, true);
    verify_value<double, double>(-5.5, -5.4, false);
}


BOOST_AUTO_TEST_CASE(masked_d0_array_d0_array)
{
    fern::MaskedConstant<int8_t> constant1;
    fern::MaskedConstant<int32_t> constant2;
    fern::MaskedConstant<bool> result_we_get;

    // MaskedConstants with non-masking equal. ---------------------------------
    // Constants are not masked.
    constant1.mask() = false;
    constant1.value() = 5;
    constant2.mask() = false;
    constant2.value() = 5;
    result_we_get.value() = false;
    result_we_get.mask() = false;
    fern::algebra::equal(constant1, constant2, result_we_get);
    BOOST_CHECK(!result_we_get.mask());
    BOOST_CHECK_EQUAL(result_we_get.value(), true);

    // Constant is masked.
    constant1.mask() = true;
    constant1.value() = 5;
    constant2.mask() = false;
    constant2.value() = 5;
    result_we_get.value() = false;
    result_we_get.mask() = false;
    fern::algebra::equal(constant1, constant2, result_we_get);
    BOOST_CHECK(!result_we_get.mask());
    BOOST_CHECK_EQUAL(result_we_get.value(), true);

    // MaskedConstant with masking equal. --------------------------------------
    using InputNoDataPolicy = fern::DetectNoDataByValue<bool>;
    using OutputNoDataPolicy = fern::MarkNoDataByValue<bool>;

    // Constants are not masked.
    constant1.mask() = false;
    constant1.value() = 5;
    constant2.mask() = false;
    constant2.value() = 5;
    result_we_get.value() = false;
    result_we_get.mask() = constant1.mask() || constant2.mask();
    fern::algebra::equal<fern::MaskedConstant<int8_t>,
        fern::MaskedConstant<int32_t>, fern::MaskedConstant<bool>,
        fern::binary::DiscardDomainErrors, fern::binary::DiscardRangeErrors,
        InputNoDataPolicy,
        OutputNoDataPolicy>(
            InputNoDataPolicy(result_we_get.mask(), true),
            OutputNoDataPolicy(result_we_get.mask(), true),
            constant1, constant2, result_we_get);
    BOOST_CHECK(!result_we_get.mask());
    BOOST_CHECK_EQUAL(result_we_get.value(), true);

    // Constants are masked.
    constant1.value() = 5;
    constant1.mask() = true;
    constant2.value() = 5;
    constant2.mask() = false;
    result_we_get.value() = false;
    result_we_get.mask() = constant1.mask() || constant2.mask();
    fern::algebra::equal<fern::MaskedConstant<int8_t>,
        fern::MaskedConstant<int32_t>, fern::MaskedConstant<bool>,
        fern::binary::DiscardDomainErrors, fern::binary::DiscardRangeErrors,
        InputNoDataPolicy,
        OutputNoDataPolicy>(
            InputNoDataPolicy(result_we_get.mask(), true),
            OutputNoDataPolicy(result_we_get.mask(), true),
            constant1, constant2, result_we_get);
    BOOST_CHECK(result_we_get.mask());
    BOOST_CHECK_EQUAL(result_we_get.value(), false);
}


BOOST_AUTO_TEST_CASE(d1_array_d1_array)
{
    // vector
    {
        std::vector<int32_t> array1{1, 2, 3};
        std::vector<uint64_t> array2{1, 4, 3};
        std::vector<bool> result(3);
        fern::algebra::equal(array1, array2, result);
        BOOST_CHECK_EQUAL(result[0], true);
        BOOST_CHECK_EQUAL(result[1], false);
        BOOST_CHECK_EQUAL(result[2], true);
    }

    // 1d array
    {
        fern::Array<uint8_t, 1> array1{1, 2, 3};
        fern::Array<uint16_t, 1> array2{1, 4, 3};
        std::vector<bool> result(3);
        fern::algebra::equal(array1, array2, result);
        BOOST_CHECK_EQUAL(result[0], true);
        BOOST_CHECK_EQUAL(result[1], false);
        BOOST_CHECK_EQUAL(result[2], true);
    }

    // empty
    {
        std::vector<int32_t> array1;
        std::vector<int32_t> array2;
        std::vector<bool> result;
        fern::algebra::equal(array1, array2, result);
        BOOST_CHECK(result.empty());
    }
}


BOOST_AUTO_TEST_CASE(masked_d1_array)
{
    using InputNoDataPolicy = fern::DetectNoDataByValue<fern::Mask<1>>;
    using OutputNoDataPolicy = fern::MarkNoDataByValue<fern::Mask<1>>;

    fern::MaskedArray<int32_t, 1> array1{1, 2, 3};
    fern::MaskedArray<int32_t, 1> array2{1, 4, 3};

    // 1d masked arrays with non-masking equal
    {
        fern::MaskedArray<bool, 1> result(3, false);
        fern::algebra::equal(array1, array2, result);
        BOOST_CHECK_EQUAL(result.mask()[0], false);
        BOOST_CHECK_EQUAL(result.mask()[1], false);
        BOOST_CHECK_EQUAL(result.mask()[2], false);
        BOOST_CHECK_EQUAL(result[0], true);
        BOOST_CHECK_EQUAL(result[1], false);
        BOOST_CHECK_EQUAL(result[2], true);
    }

    // 1d masked arrays with masking equal
    {
        fern::MaskedArray<bool, 1> result(3, false);
        result.mask()[2] = true;
        fern::algebra::equal<
            fern::MaskedArray<int32_t, 1>,
            fern::MaskedArray<int32_t, 1>,
            fern::MaskedArray<bool, 1>,
            fern::binary::DiscardDomainErrors,
            fern::binary::DiscardRangeErrors,
            InputNoDataPolicy,
            OutputNoDataPolicy>(
                InputNoDataPolicy(result.mask(), true),
                OutputNoDataPolicy(result.mask(), true),
                array1, array2, result);
        BOOST_CHECK_EQUAL(result.mask()[0], false);
        BOOST_CHECK_EQUAL(result.mask()[1], false);
        BOOST_CHECK_EQUAL(result.mask()[2], true);
        BOOST_CHECK_EQUAL(result[0], true);
        BOOST_CHECK_EQUAL(result[1], false);
        BOOST_CHECK_EQUAL(result[2], false);  // <-- Not touched.

        // Mask the whole input. Result must be masked too.
        result.mask_all();
        fern::algebra::equal<
            fern::MaskedArray<int32_t, 1>,
            fern::MaskedArray<int32_t, 1>,
            fern::MaskedArray<bool, 1>,
            fern::binary::DiscardDomainErrors,
            fern::binary::DiscardRangeErrors,
            InputNoDataPolicy,
            OutputNoDataPolicy>(
                InputNoDataPolicy(result.mask(), true),
                OutputNoDataPolicy(result.mask(), true),
                array1, array2, result);
        BOOST_CHECK_EQUAL(result.mask()[0], true);
        BOOST_CHECK_EQUAL(result.mask()[1], true);
        BOOST_CHECK_EQUAL(result.mask()[2], true);
    }

    // empty
    {
        fern::MaskedArray<int32_t, 1> empty_array;
        fern::MaskedArray<bool, 1> result;
        fern::algebra::equal<
            fern::MaskedArray<int32_t, 1>,
            fern::MaskedArray<int32_t, 1>,
            fern::MaskedArray<bool, 1>,
            fern::binary::DiscardDomainErrors,
            fern::binary::DiscardRangeErrors,
            InputNoDataPolicy,
            OutputNoDataPolicy>(
                InputNoDataPolicy(result.mask(), true),
                OutputNoDataPolicy(result.mask(), true),
                empty_array, empty_array, result);
        BOOST_CHECK(result.empty());
    }
}


BOOST_AUTO_TEST_CASE(d2_array_d2_array)
{
    // 2d array
    {
        fern::Array<int8_t, 2> array1{
            { -2, -1 },
            {  0,  9 },
            {  1,  2 }
        };
        fern::Array<int8_t, 2> array2{
            { -2, -1 },
            {  5,  9 },
            {  1,  2 }
        };
        fern::Array<bool, 2> result{
            { false, false },
            { false, false },
            { false, false }
        };
        fern::algebra::equal(array1, array2, result);
        BOOST_CHECK_EQUAL(result[0][0], true);
        BOOST_CHECK_EQUAL(result[0][1], true);
        BOOST_CHECK_EQUAL(result[1][0], false);
        BOOST_CHECK_EQUAL(result[1][1], true);
        BOOST_CHECK_EQUAL(result[2][0], true);
        BOOST_CHECK_EQUAL(result[2][1], true);
    }
}


BOOST_AUTO_TEST_CASE(masked_d2_array_d2_array)
{
    fern::MaskedArray<int8_t, 2> array1{
        { -2, -1 },
        {  0,  9 },
        {  1,  2 }
    };
    fern::MaskedArray<int8_t, 2> array2{
        { -2, -1 },
        {  5,  9 },
        {  1,  2 }
    };

    // 2d masked array with non-masking equal
    {
        fern::MaskedArray<bool, 2> result{
            { true, true },
            { true, true },
            { true, true }
        };
        fern::algebra::equal(array1, array2, result);
        BOOST_CHECK_EQUAL(result[0][0], true);
        BOOST_CHECK_EQUAL(result[0][1], true);
        BOOST_CHECK_EQUAL(result[1][0], false);
        BOOST_CHECK_EQUAL(result[1][1], true);
        BOOST_CHECK_EQUAL(result[2][0], true);
        BOOST_CHECK_EQUAL(result[2][1], true);
    }

    // 2d masked arrays with masking equal
    {
        // TODO
    }
}


BOOST_AUTO_TEST_CASE(concurrent)
{
    // Create a somewhat larger array.
    size_t const nr_rows = 6000;
    size_t const nr_cols = 4000;
    auto const extents = fern::extents[nr_rows][nr_cols];
    fern::Array<int32_t, 2> array1(extents);
    fern::Array<int32_t, 2> array2(extents);
    fern::Array<bool, 2> result(extents);
    fern::algebra::Equal<
        fern::Array<int32_t, 2>,
        fern::Array<int32_t, 2>,
        fern::Array<bool, 2>> equal;

    std::iota(array1.data(), array1.data() + array1.num_elements(), 0);
    std::iota(array2.data(), array2.data() + array2.num_elements(), 0);

    // Serial.
    {
        result.fill(false);
        fern::serial::execute(equal, array1, array2, result);
        BOOST_CHECK(std::all_of(result.data(), result.data() +
            result.num_elements(), [](bool equal){ return equal; }));
    }

    // Concurrent.
    {
        result.fill(false);
        fern::ThreadClient client;
        fern::concurrent::execute(equal, array1, array2, result);
        BOOST_CHECK(std::all_of(result.data(), result.data() +
            result.num_elements(), [](bool equal){ return equal; }));
    }

    {
        using InputNoDataPolicy = fern::SkipNoData;
        using OutputNoDataPolicy = fern::MarkNoDataByValue<fern::Mask<2>>;

        fern::MaskedArray<bool, 2> result(extents);
        fern::algebra::Equal<
            fern::Array<int32_t, 2>,
            fern::Array<int32_t, 2>,
            fern::MaskedArray<bool, 2>,
            fern::binary::DiscardDomainErrors,
            fern::binary::DiscardRangeErrors,
            InputNoDataPolicy,
            OutputNoDataPolicy> equal(
                InputNoDataPolicy(),
                OutputNoDataPolicy(result.mask(), true));

        // Verify executor can handle masked result.
        fern::ThreadClient client;
        fern::concurrent::execute(equal, array1, array2, result);
        BOOST_CHECK(std::all_of(result.data(), result.data() +
            result.num_elements(), [](bool equal){ return equal; }));
    }
}

// TODO Test 1d array == 0d array
//      Test 2d array == 1d array
//      etc

BOOST_AUTO_TEST_SUITE_END()
