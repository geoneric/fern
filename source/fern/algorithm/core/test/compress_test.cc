#define BOOST_TEST_MODULE fern algorithm core compress
#include <boost/test/unit_test.hpp>
#include "fern/core/constant_traits.h"
#include "fern/core/vector_traits.h"
// #include "fern/feature/core/masked_array_traits.h"
// #include "fern/algorithm/algebra/boole/and.h"
#include "fern/algorithm/core/compress.h"
#include "fern/algorithm/core/test/test_utils.h"


namespace fa = fern::algorithm;


BOOST_FIXTURE_TEST_SUITE(compress, fern::ThreadClient)

void test_array_1d(
    fa::ExecutionPolicy const& execution_policy)
{
    size_t const nr_threads{fern::ThreadClient::hardware_concurrency()};
    size_t const nr_elements{10 * nr_threads};
    std::vector<int> values(nr_elements);
    std::vector<int> result_we_want(nr_elements);
    std::vector<int> result_we_got(nr_elements);
    int const no_data_value{99};

    // 0, 1, 2, 3, ..., n-1
    // All values dividable by 5 are replaced by a special value.
    std::iota(values.begin(), values.end(), 0);
    std::transform(values.begin(), values.end(), values.begin(),
        [&](int const& value) {
            return value % 5 == 0? no_data_value : value; });

    // Compression without input no-data is the same as copy.
    result_we_got = values;

    size_t count{12345};

    {
        fa::core::compress(execution_policy, values, result_we_got, count);
        BOOST_CHECK_EQUAL(count, values.size());
        BOOST_CHECK(result_we_got == values);
    }
}


BOOST_AUTO_TEST_CASE(array_1d_sequential)
{
    test_array_1d(fa::sequential);
}


BOOST_AUTO_TEST_CASE(array_1d_parallel)
{
    test_array_1d(fa::parallel);
}


void test_array_1d_masked(
    fa::ExecutionPolicy const& execution_policy)
{
    size_t const nr_threads{fern::ThreadClient::hardware_concurrency()};
    size_t const nr_elements{10 * nr_threads};
    std::vector<int> values(nr_elements);
    std::vector<int> result_we_want(nr_elements);
    std::vector<int> result_we_got(nr_elements);
    int const no_data_value{99};

    // 0, 1, 2, 3, ..., n-1
    // All values dividable by 5 are replaced by a special value.
    std::iota(values.begin(), values.end(), 0);
    std::transform(values.begin(), values.end(), values.begin(),
        [&](int const& value) {
            return value % 5 == 0? no_data_value : value; });

    using InputNoDataPolicy = fa::DetectNoDataByValue<std::vector<int>>;
    InputNoDataPolicy input_no_data_policy(values, no_data_value);

    result_we_want = values;
    result_we_want.erase(std::remove(result_we_want.begin(),
        result_we_want.end(), no_data_value), result_we_want.end());

    size_t count{999999};

    {
        fa::core::compress(input_no_data_policy,
            execution_policy, values, result_we_got, count);
        result_we_got.resize(count);
        BOOST_CHECK_EQUAL(count, result_we_want.size());
        BOOST_CHECK(result_we_got == result_we_want);
    }
}


BOOST_AUTO_TEST_CASE(array_1d_masked_sequential)
{
    test_array_1d_masked(fa::sequential);
}


BOOST_AUTO_TEST_CASE(array_1d_masked_parallel)
{
    test_array_1d_masked(fa::parallel);
}


void test_array_2d(
    fa::ExecutionPolicy const& execution_policy)
{
    // TODO

    // Create 2D array of values.
    // Compression should result in the same array.
}


BOOST_AUTO_TEST_CASE(array_2d_sequential)
{
    test_array_2d(fa::sequential);
}


BOOST_AUTO_TEST_CASE(array_2d_parallel)
{
    test_array_2d(fa::parallel);
}


void test_array_2d_masked(
    fa::ExecutionPolicy const& execution_policy)
{
    // TODO
}


BOOST_AUTO_TEST_CASE(array_2d_masked_sequential)
{
    test_array_2d_masked(fa::sequential);
}


BOOST_AUTO_TEST_CASE(array_2d_masked_parallel)
{
    test_array_2d_masked(fa::parallel);
}







// template<
//     class Value1,
//     class Value2,
//     class Result>
// void verify_value(
//     Value1 const& value1,
//     Value2 const& value2,
//     Result const& result_we_want)
// {
//     Result result_we_get;
//     fa::core::compress<>(fa::sequential, value1, value2, result_we_get);
//     BOOST_CHECK_EQUAL(result_we_get, result_we_want);
// }
// 
// 
// BOOST_AUTO_TEST_CASE(algorithm)
// {
//     {
//         verify_value<>( 4, 5, 4);
//         verify_value<>( 5, 4, 5);
//     }
// }
// 
// 
// BOOST_AUTO_TEST_CASE(array_2d_masked_0d)
// {
//     size_t const nr_rows{3};
//     size_t const nr_cols{3};
//     fern::MaskedArray<int, 2> value1(fern::extents[nr_rows][nr_cols]);
//     int const value2{3};
//     fern::MaskedArray<int, 2> result_we_got(fern::extents[nr_rows][nr_cols]);
// 
//     fa::DetectNoDataByValue<fern::Mask<2>> value1_input_no_data_policy(
//         value1.mask(), true);
//     fa::SkipNoData<> value2_input_no_data_policy;
// 
//     fa::DetectNoDataByValue<fern::Mask<2>,
//         fa::DetectNoDataByValue<fern::Mask<2>>,
//         fa::SkipNoData<>> input_no_data_policy(result_we_got.mask(), true,
//             std::move(value1_input_no_data_policy),
//             std::move(value2_input_no_data_policy));
//     fa::MarkNoDataByValue<fern::Mask<2>> output_no_data_policy(
//         result_we_got.mask(), true);
// 
// 
//     value1[1][2] = 4;
//     value1[2][1] = 5;
//     value1[2][2] = -1;
//     value1.mask().fill(true);
//     value1.mask()[1][2] = false;
//     value1.mask()[2][1] = false;
//     value1.mask()[2][2] = false;
// 
//     fern::MaskedArray<int, 2> result_we_want(fern::extents[nr_rows][nr_cols]);
//     result_we_want[0][0] = value2;
//     result_we_want[0][1] = value2;
//     result_we_want[0][2] = value2;
//     result_we_want[1][0] = value2;
//     result_we_want[1][1] = value2;
//     result_we_want[1][2] = value1[1][2];
//     result_we_want[2][0] = value2;
//     result_we_want[2][1] = value1[2][1];
//     result_we_want[2][2] = value1[2][2];
// 
//     auto const& execution_policy(fa::sequential);
// 
//     fa::core::compress(input_no_data_policy, output_no_data_policy,
//         execution_policy, value1, value2, result_we_got);
// 
//     BOOST_CHECK(fern::compare(execution_policy, result_we_got,
//         result_we_want));
// }
// 
// 
// BOOST_AUTO_TEST_CASE(array_2d_masked_2d)
// {
//     size_t const nr_rows{3};
//     size_t const nr_cols{3};
//     fern::MaskedArray<int, 2> value1(fern::extents[nr_rows][nr_cols]);
//     fern::MaskedArray<int, 2> value2(fern::extents[nr_rows][nr_cols]);
//     fern::MaskedArray<int, 2> value3(fern::extents[nr_rows][nr_cols]);
//     fern::MaskedArray<int, 2> result_we_got1(fern::extents[nr_rows][nr_cols]);
//     fern::MaskedArray<int, 2> result_we_got2(fern::extents[nr_rows][nr_cols]);
// 
//     // +---+---+---+
//     // | X | X | X |
//     // +---+---+---+
//     // | X | X | 4 |
//     // +---+---+---+
//     // | X | X | 1 |
//     // +---+---+---+
//     value1[1][2] = 4;
//     value1[2][2] = 1;
//     value1.mask().fill(true);
//     value1.mask()[1][2] = false;
//     value1.mask()[2][2] = false;
// 
//     // +---+---+---+
//     // | 0 | X | X |
//     // +---+---+---+
//     // | X | X | 8 |
//     // +---+---+---+
//     // | X | 2 | X |
//     // +---+---+---+
//     value2[0][0] = 0;
//     value2[1][2] = 8;
//     value2[2][1] = 2;
//     value2.mask().fill(true);
//     value2.mask()[0][0] = false;
//     value2.mask()[1][2] = false;
//     value2.mask()[2][1] = false;
// 
//     // +---+---+---+
//     // | 4 | 4 | 4 |
//     // +---+---+---+
//     // | 4 | 4 | 4 |
//     // +---+---+---+
//     // | X | 4 | 4 |
//     // +---+---+---+
//     value3.fill(4);
//     value3[2][0] = -9;
//     value3.mask().fill(false);
//     value3.mask()[2][0] = true;
// 
//     // +---+---+---+
//     // | 0 | 4 | 4 |
//     // +---+---+---+
//     // | 4 | 4 | 8 |
//     // +---+---+---+
//     // | X | 2 | 1 |
//     // +---+---+---+
//     fern::MaskedArray<int, 2> result_we_want(fern::extents[nr_rows][nr_cols]);
//     result_we_want[0][0] = value2[0][0];
//     result_we_want[0][1] = value3[0][1];
//     result_we_want[0][2] = value3[0][2];
//     result_we_want[1][0] = value3[1][0];
//     result_we_want[1][1] = value3[1][1];
//     result_we_want[1][2] = value1[1][2];
//     result_we_want[2][0] = 0;
//     result_we_want[2][1] = value2[2][1];
//     result_we_want[2][2] = value1[2][2];
//     result_we_want.mask().fill(false);
//     result_we_want.mask()[2][0] = true;
// 
// 
//     auto const& execution_policy(fa::sequential);
// 
//     fa::MarkNoDataByValue<fern::Mask<2>> output_no_data_policy(
//         result_we_got1.mask(), true);
// 
//     // result1 = compress(value1, value2)
//     {
//         fa::DetectNoDataByValue<fern::Mask<2>,
//             fa::DetectNoDataByValue<fern::Mask<2>>,
//             fa::DetectNoDataByValue<fern::Mask<2>>> input_no_data_policy(
//                 result_we_got1.mask(), true,
//                 fa::DetectNoDataByValue<fern::Mask<2>>(value1.mask(), true),
//                 fa::DetectNoDataByValue<fern::Mask<2>>(value2.mask(), true));
// 
//         fa::algebra::and_(execution_policy, value1.mask(), value2.mask(),
//             result_we_got1.mask());
// 
//         fa::core::compress(input_no_data_policy, output_no_data_policy,
//             execution_policy, value1, value2, result_we_got1);
// 
//     }
// 
//     // result2 = compress(result1, value3)
//     {
//         fa::DetectNoDataByValue<fern::Mask<2>,
//             fa::DetectNoDataByValue<fern::Mask<2>>,
//             fa::DetectNoDataByValue<fern::Mask<2>>> input_no_data_policy(
//                 result_we_got2.mask(), true,
//                 fa::DetectNoDataByValue<fern::Mask<2>>(result_we_got1.mask(),
//                     true),
//                 fa::DetectNoDataByValue<fern::Mask<2>>(value3.mask(), true));
// 
//         fa::algebra::and_(execution_policy, result_we_got1.mask(),
//             value3.mask(), result_we_got2.mask());
// 
//         fa::core::compress(input_no_data_policy, output_no_data_policy,
//             execution_policy, result_we_got1, value3, result_we_got2);
//     }
// 
//     BOOST_CHECK(fern::compare(execution_policy, result_we_got2,
//         result_we_want));
// }

BOOST_AUTO_TEST_SUITE_END()
