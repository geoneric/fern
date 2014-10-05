#define BOOST_TEST_MODULE fern algorithm convolution convolve
#include <boost/test/unit_test.hpp>
#include "fern/core/constant_traits.h"
#include "fern/core/types.h"
#include "fern/feature/core/masked_array_traits.h"
#include "fern/algorithm/algebra/elementary/equal.h"
#include "fern/algorithm/statistic/count.h"
#include "fern/algorithm/convolution/neighborhood/square.h"
#include "fern/algorithm/convolution/neighborhood/square_traits.h"
#include "fern/algorithm/convolution/convolve.h"
#include "fern/algorithm/convolution/replace_no_data_by_focal_average.h"


BOOST_FIXTURE_TEST_SUITE(convolve, fern::ThreadClient)

namespace fa = fern::algorithm;


void compare_result1(
    fern::Array<double, 2> const& result)
{
    // Compare results.
    // Upper left corner. ------------------------------------------------------
    // 0, 0:
    // +----+----+----+
    // |  0 |  1 |  2 |
    // +----+----+----+
    // |  6 |  7 |  8 |
    // +----+----+----+
    // | 12 | 13 | 14 |
    // +----+----+----+
    BOOST_CHECK_CLOSE(result[0][0], 63.0 / 9.0, 1e-6);

    // 0, 1:
    // +----+----+----+----+
    // |  0 |  1 |  2 |  3 |
    // +----+----+----+----+
    // |  6 |  7 |  8 |  9 |
    // +----+----+----+----+
    // | 12 | 13 | 14 | 15 |
    // +----+----+----+----+
    BOOST_CHECK_CLOSE(result[0][1], 90.0 / 12.0, 1e-6);

    // 1, 1:
    // +----+----+----+----+
    // |  0 |  1 |  2 |  3 |
    // +----+----+----+----+
    // |  6 |  7 |  8 |  9 |
    // +----+----+----+----+
    // | 12 | 13 | 14 | 15 |
    // +----+----+----+----+
    // | 18 | 19 | 20 | 21 |
    // +----+----+----+----+
    BOOST_CHECK_CLOSE(result[1][1], 168.0 / 16.0, 1e-6);

    // Upper right corner. -----------------------------------------------------
    // 0, 4:
    // +----+----+----+----+
    // |  2 |  3 |  4 |  5 |
    // +----+----+----+----+
    // |  8 |  9 | 10 | 11 |
    // +----+----+----+----+
    // | 14 | 15 | 16 | 17 |
    // +----+----+----+----+
    BOOST_CHECK_CLOSE(result[0][4], 114.0 / 12.0, 1e-6);

    // Lower left corner. ------------------------------------------------------
    // 6, 1:
    // +----+----+----+----+
    // | 24 | 25 | 26 | 27 |
    // +----+----+----+----+
    // | 30 | 31 | 32 | 33 |
    // +----+----+----+----+
    // | 36 | 37 | 38 | 39 |
    // +----+----+----+----+
    BOOST_CHECK_CLOSE(result[6][1], 378 / 12.0, 1e-6);

    // Lower right corner. -----------------------------------------------------
    // 6, 5
    // +----+----+----+
    // | 27 | 28 | 29 |
    // +----+----+----+
    // | 33 | 34 | 35 |
    // +----+----+----+
    // | 39 | 40 | 41 |
    // +----+----+----+
    BOOST_CHECK_CLOSE(result[6][5], 306 / 9.0, 1e-6);

    // North side. -------------------------------------------------------------
    // 0, 2
    // +----+----+----+----+----+
    // |  0 |  1 |  2 |  3 |  4 |
    // +----+----+----+----+----+
    // |  6 |  7 |  8 |  9 | 10 |
    // +----+----+----+----+----+
    // | 12 | 13 | 14 | 15 | 16 |
    // +----+----+----+----+----+
    BOOST_CHECK_CLOSE(result[0][2], 120 / 15.0, 1e-6);

    // West side. --------------------------------------------------------------
    // 4, 0
    // +----+----+----+
    // | 12 | 13 | 14 |
    // +----+----+----+
    // | 18 | 19 | 20 |
    // +----+----+----+
    // | 24 | 25 | 26 |
    // +----+----+----+
    // | 30 | 31 | 32 |
    // +----+----+----+
    // | 36 | 37 | 38 |
    // +----+----+----+
    BOOST_CHECK_CLOSE(result[4][0], 375 / 15.0, 1e-6);

    // East side. --------------------------------------------------------------
    // 2, 4
    // +----+----+----+----+
    // |  2 |  3 |  4 |  5 |
    // +----+----+----+----+
    // |  8 |  9 | 10 | 11 |
    // +----+----+----+----+
    // | 14 | 15 | 16 | 17 |
    // +----+----+----+----+
    // | 20 | 21 | 22 | 23 |
    // +----+----+----+----+
    // | 26 | 27 | 28 | 29 |
    // +----+----+----+----+
    BOOST_CHECK_CLOSE(result[2][4], 310 / 20.0, 1e-6);

    // South side.
    // 6, 3
    // +----+----+----+----+----+
    // | 25 | 26 | 27 | 28 | 29 |
    // +----+----+----+----+----+
    // | 31 | 32 | 33 | 34 | 35 |
    // +----+----+----+----+----+
    // | 37 | 38 | 39 | 40 | 41 |
    // +----+----+----+----+----+
    BOOST_CHECK_CLOSE(result[6][3], 495 / 15.0, 1e-6);

    // Inner part.
    // 3, 2
    // +----+----+----+----+----+
    // |  6 |  7 |  8 |  9 | 10 |
    // +----+----+----+----+----+
    // | 12 | 13 | 14 | 15 | 16 |
    // +----+----+----+----+----+
    // | 18 | 19 | 20 | 21 | 22 |
    // +----+----+----+----+----+
    // | 24 | 25 | 26 | 27 | 28 |
    // +----+----+----+----+----+
    // | 30 | 31 | 32 | 33 | 34 |
    // +----+----+----+----+----+
    BOOST_CHECK_CLOSE(result[3][2], 500 / 25.0, 1e-6);

    // Make sure all cells in the result have a new value.
    BOOST_CHECK_EQUAL(std::count(result.data(), result.data() +
        result.num_elements(), 0), 0);
}


void compare_result2(
    fern::Array<double, 2> const& result)
{
    // Upper left corner. ----------------------------------------------
    BOOST_CHECK_CLOSE(result[0][0], (63.0 + 24.5) / (9.0 + 7.0), 1e-6);
    BOOST_CHECK_CLOSE(result[0][1], (90.0 + 27.5) / (12.0 + 8.0), 1e-6);
    BOOST_CHECK_CLOSE(result[1][1], (168.0 + 45.5) / (16.0 + 9.0),
        1e-6);

    // Upper right corner. ---------------------------------------------
    BOOST_CHECK_CLOSE(result[0][4], (114.0 + 54.5) / (12.0 + 8.0),
        1e-6);

    // Lower left corner. ----------------------------------------------
    // Out of image values:
    //     (18 + 24 + 30) / 3 ->  72 / 3
    //     (24 + 30 + 36) / 3 ->  90 / 3
    //     (30 + 36     ) / 2 ->  66 / 2
    //     (36          ) / 1 ->  36 / 1
    //     (36 + 37     ) / 2 ->  73 / 2
    //     (36 + 37 + 38) / 3 -> 111 / 3
    //     (37 + 38 + 39) / 3 -> 114 / 3
    //     (38 + 39 + 40) / 3 -> 117 / 3
    BOOST_CHECK_CLOSE(result[6][1], (378.0 + 273.5) / (12.0 + 8.0),
        1e-6);

    // Lower right corner. ---------------------------------------------
    // Out of image values:
    //     (38 + 39 + 40) / 3 -> 117 / 3
    //     (39 + 40 + 41) / 3 -> 120 / 3
    //     (40 + 41     ) / 2 ->  81 / 2
    //     (41          ) / 1 ->  41 / 1
    //     (41 + 35     ) / 2 ->  76 / 2
    //     (29 + 35 + 41) / 3 -> 105 / 3
    //     (23 + 29 + 35) / 3 ->  87 / 3
    BOOST_CHECK_CLOSE(result[6][5], (306.0 + 262.5) / (9.0 + 7.0),
        1e-6);

    // North side. -----------------------------------------------------
    // Out of image values:
    // (0 + 1    ) / 2
    // (0 + 1 + 2) / 3
    // (1 + 2 + 3) / 3
    // (2 + 3 + 4) / 3
    // (3 + 4 + 5) / 3
    BOOST_CHECK_CLOSE(result[0][2], (120.0 + 10.5) / (15.0 + 5.0),
        1e-6);

    // West side. ------------------------------------------------------
    // Out of image values:
    // ( 6 + 12 + 18) / 3 -> 36 / 3
    // (12 + 18 + 24) / 3 -> 54 / 3
    // (18 + 24 + 30) / 3 -> 72 / 3
    // (24 + 30 + 36) / 3 -> 90 / 3
    // (30 + 36     ) / 2 -> 66 / 2
    BOOST_CHECK_CLOSE(result[4][0], (375.0 + 117.0) / (15.0 + 5.0),
        1e-6);

    // East side. ------------------------------------------------------
    // Out of image values:
    // ( 5 + 11     ) / 2
    // ( 5 + 11 + 17) / 3
    // (11 + 17 + 23) / 3
    // (17 + 23 + 29) / 3
    // (23 + 29 + 35) / 3
    BOOST_CHECK_CLOSE(result[2][4], (310.0 + 88.0) / (20.0 + 5.0),
        1e-6);

    // South side. -----------------------------------------------------
    // Out of image values:
    // (36 + 37 + 38) / 3
    // (37 + 38 + 39) / 3
    // (38 + 39 + 40) / 3
    // (39 + 40 + 41) / 3
    // (40 + 41     ) / 2
    BOOST_CHECK_CLOSE(result[6][3], (495.0 + 194.5) / (15.0 + 5.0),
        1e-6);

    // Inner part. -----------------------------------------------------
    // No out of image values.
    BOOST_CHECK_CLOSE(result[3][2], 500 / 25.0, 1e-6);

    // Make sure all cells in the result have a new value.
    BOOST_CHECK_EQUAL(std::count(result.data(), result.data() +
        result.num_elements(), 0), 0);
}


BOOST_AUTO_TEST_CASE(convolve)
{
    // Kernel with radius 2.
    {
        // Create input array:
        // +----+----+----+----+----+----+
        // |  0 |  1 |  2 |  3 |  4 |  5 |
        // +----+----+----+----+----+----+
        // |  6 |  7 |  8 |  9 | 10 | 11 |
        // +----+----+----+----+----+----+
        // | 12 | 13 | 14 | 15 | 16 | 17 |
        // +----+----+----+----+----+----+
        // | 18 | 19 | 20 | 21 | 22 | 23 |
        // +----+----+----+----+----+----+
        // | 24 | 25 | 26 | 27 | 28 | 29 |
        // +----+----+----+----+----+----+
        // | 30 | 31 | 32 | 33 | 34 | 35 |
        // +----+----+----+----+----+----+
        // | 36 | 37 | 38 | 39 | 40 | 41 |
        // +----+----+----+----+----+----+
        size_t const nr_rows = 7;
        size_t const nr_cols = 6;
        auto extents = fern::extents[nr_rows][nr_cols];
        fern::Array<double, 2> argument(extents);
        std::iota(argument.data(), argument.data() + argument.num_elements(),
            0);

        // Calculate local average.
        // Define kernel shape and weights.
        fern::Square<int, 2> kernel({
            {1, 1, 1, 1, 1},
            {1, 1, 1, 1, 1},
            {1, 1, 1, 1, 1},
            {1, 1, 1, 1, 1},
            {1, 1, 1, 1, 1}
        });

        // Convolute while skipping out-of-image cells.
        {
            // Sequential.
            {
                fern::Array<double, 2> result(extents);
                fa::convolution::convolve(fa::sequential, argument, kernel,
                    result);
                compare_result1(result);
            }

            // Parallel.
            {
                fern::Array<double, 2> result(extents);
                fa::convolution::convolve(fa::parallel, argument, kernel,
                    result);
                compare_result1(result);
            }
        }

        // Convolute while calculating values for out-of-image cells.
        {
            using AlternativeForNoDataPolicy=fa::convolve::SkipNoData;
            using NormalizePolicy=fa::convolve::DivideByWeights;
            using OutOfImagePolicy=
                fa::convolve::ReplaceOutOfImageByFocalAverage;
            using InputNoDataPolicy=fa::SkipNoData<>;
            using OutputNoDataPolicy=fa::DontMarkNoData;

            // Sequential.
            {
                fern::Array<double, 2> result(extents);
                fa::convolution::convolve<
                    AlternativeForNoDataPolicy,
                    NormalizePolicy,
                    OutOfImagePolicy,
                    fa::unary::DiscardRangeErrors,
                    InputNoDataPolicy,
                    OutputNoDataPolicy>(fa::sequential, argument, kernel,
                        result);
                compare_result2(result);
            }

            // Parallel.
            {
                fern::Array<double, 2> result(extents);
                fa::convolution::convolve<
                    AlternativeForNoDataPolicy,
                    NormalizePolicy,
                    OutOfImagePolicy,
                    fa::unary::DiscardRangeErrors,
                    InputNoDataPolicy,
                    OutputNoDataPolicy>(fa::parallel, argument, kernel,
                        result);
                compare_result2(result);
            }
        }
    }

    // Kernel with radius 1.
    // This used to crash.
    {
        // Create input array:
        // +----+----+----+----+----+----+
        // |  0 |  1 |  2 |  3 |  4 |  5 |
        // +----+----+----+----+----+----+
        // |  6 |  7 |  8 |  9 | 10 | 11 |
        // +----+----+----+----+----+----+
        // | 12 | 13 | 14 | 15 | 16 | 17 |
        // +----+----+----+----+----+----+
        // | 18 | 19 | 20 | 21 | 22 | 23 |
        // +----+----+----+----+----+----+
        // | 24 | 25 | 26 | 27 | 28 | 29 |
        // +----+----+----+----+----+----+
        // | 30 | 31 | 32 | 33 | 34 | 35 |
        // +----+----+----+----+----+----+
        // | 36 | 37 | 38 | 39 | 40 | 41 |
        // +----+----+----+----+----+----+
        size_t const nr_rows = 7;
        size_t const nr_cols = 6;
        auto extents = fern::extents[nr_rows][nr_cols];
        fern::Array<double, 2> argument(extents);
        std::iota(argument.data(), argument.data() + argument.num_elements(),
            0);

        // Calculate local average.
        // Define kernel shape and weights.
        fern::Square<int, 1> kernel({
            {1, 1, 1},
            {1, 1, 1},
            {1, 1, 1}
        });

        // Sequential.
        {
            fern::Array<double, 2> result(extents);
            fa::convolution::convolve(fa::sequential, argument, kernel,
                result);
        }

        // Parallel.
        {
            fern::Array<double, 2> result(extents);
            fa::convolution::convolve(fa::parallel, argument, kernel,
                result);
        }
    }
}


// TODO Test Small image with out of image policy with larger kernel.
// TODO Test Focal Sum, with kernel that doesn't weigh.


template<
    class Value,
    class Result>
using OutOfRangePolicy = fa::convolve::OutOfRangePolicy<Value, Result>;


BOOST_AUTO_TEST_CASE(out_of_range_policy)
{
    // Make sure that out of range can be detected and that no-data can be
    // written when it happens.

    {
        auto min_float32 = fern::min<fern::f32>();
        auto max_float32 = fern::max<fern::f32>();

        OutOfRangePolicy<fern::f32, fern::f32> policy;
        BOOST_CHECK(policy.within_range(5.0));
        BOOST_CHECK(policy.within_range(-5.0));
        BOOST_CHECK(policy.within_range(0.0));
        BOOST_CHECK(policy.within_range(min_float32));
        BOOST_CHECK(policy.within_range(max_float32));
        BOOST_CHECK(!policy.within_range(2 * max_float32));
    }
}


BOOST_AUTO_TEST_CASE(no_data_policies)
{
    // Make sure that input no-data is detected and handled correctly.
    using InputNoDataPolicy = fa::DetectNoDataByValue<fern::Mask<2>>;
    using OutputNoDataPolicy = fa::MarkNoDataByValue<fern::Mask<2>>;

    size_t const nr_rows = 3;
    size_t const nr_cols = 3;
    auto extents = fern::extents[nr_rows][nr_cols];

    // Local average kernel.
    fern::Square<int, 1> kernel_1({
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1}
    });

    fern::Square<int, 1> kernel_2({
        {2, 2, 2},
        {2, 2, 2},
        {2, 2, 2}
    });


    // Source image with a few masked values.
    {
        // +---+---+---+
        // | 0 | 1 | 2 |
        // +---+---+---+
        // | 3 | X | 5 |
        // +---+---+---+
        // | X | 7 | 8 |
        // +---+---+---+
        fern::MaskedArray<double, 2> source(extents);
        std::iota(source.data(), source.data() + source.num_elements(), 0);
        source.mask()[1][1] = true;
        source.mask()[2][0] = true;
        fern::MaskedArray<double, 2> destination(extents);

        // Skip no-data.
        {
            fern::MaskedArray<double, 2> destination(extents);
            destination.fill(999.9);
            OutputNoDataPolicy output_no_data_policy(destination.mask(), true);
            fa::convolution::convolve<
                fa::convolve::SkipNoData,
                fa::convolve::DivideByWeights,
                fa::convolve::SkipOutOfImage,
                fa::unary::DiscardRangeErrors>(
                    InputNoDataPolicy(source.mask(), true),
                    output_no_data_policy,
                    fa::sequential,
                    source, kernel_1, destination);

            // Verify mask.
            uint64_t nr_masked_cells{0};
            fa::statistic::count(fa::sequential, destination.mask(),
                true, nr_masked_cells);
            BOOST_CHECK_EQUAL(nr_masked_cells, 2u);
            BOOST_CHECK(destination.mask()[1][1]);
            BOOST_CHECK(destination.mask()[2][0]);

            // Verify values.
            fern::Array<double, 2> result_we_want({
                { 4.0/3.0, 11.0/5.0,  8.0/3.0},
                {11.0/4.0,    999.9, 23.0/5.0},
                {   999.9, 23.0/4.0, 20.0/3.0}
            });
            fern::Array<bool, 2> equal_cells(extents);
            fa::algebra::equal(fa::sequential, destination, result_we_want,
                equal_cells);

            uint64_t nr_equal_cells{0};
            fa::statistic::count(fa::sequential, equal_cells, true,
                nr_equal_cells);
            BOOST_CHECK_EQUAL(nr_equal_cells, 9u);
        }

        // TODO Remove this, replacing no-data is pre-processing.
        /// // Replace no-data by focal average.
        /// {
        ///     // Focal average of the no-data cells is
        ///     // - 26/7
        ///     // - 10/2
        ///     // +-----+--------+---+
        ///     // |  0  |     1  | 2 |
        ///     // +-----+--------+---+
        ///     // |  3  | (26/7) | 5 |
        ///     // +-----+--------+---+
        ///     // | (5) |     7  | 8 |
        ///     // +-----+--------+---+
        ///     fern::MaskedArray<double, 2> destination(extents);
        ///     destination.fill(999.9);
        ///     fa::convolution::convolve<
        ///         fa::convolve::DivideByWeights,
        ///         fa::convolve::SkipOutOfImage,
        ///         fern::unary::DiscardRangeErrors>(
        ///         fa::convolve::ReplaceNoDataByFocalAverage,
        ///             fa::sequential,
        ///             InputNoDataPolicy(source.mask(), true),
        ///             OutputNoDataPolicy(destination.mask(), true),
        ///             source, kernel_1, destination);

        ///     // Verify mask.
        ///     size_t nr_masked_cells{0};
        ///     fa::statistic::count(destination.mask(), true, nr_masked_cells);
        ///     BOOST_CHECK_EQUAL(nr_masked_cells, 2);
        ///     BOOST_CHECK(destination.mask()[1][1]);
        ///     BOOST_CHECK(destination.mask()[2][0]);

        ///     // Verify values.
        ///     double const v1 = 26.0 / 7.0;
        ///     double const v2 = 10.0 / 2.0;
        ///     fern::Array<double, 2> result_we_want({
        ///         {       (4.0 + v1)/4.0,      (11.0 + v1)/6.0,   (8.0 + v1)/4.0},
        ///         { (11.0 + v1 + v2)/6.0,                999.9,  (23.0 + v1)/6.0},
        ///         {                999.9, (23.0 + v1 + v2)/6.0,  (20.0 + v1)/4.0}
        ///     });

        ///     fern::Array<bool, 2> equal_cells(extents);
        ///     fa::algebra::equal(destination, result_we_want, equal_cells);

        ///     size_t nr_equal_cells{0};
        ///     fa::statistic::count(equal_cells, true, nr_equal_cells);
        ///     BOOST_CHECK_EQUAL(nr_equal_cells, 9);
        /// }
    }


    // Source image with lots of masked values.
    {
        fern::MaskedArray<double, 2> source(extents);
        std::iota(source.data(), source.data() + source.num_elements(), 0);
        source.mask_all();
        fern::MaskedArray<double, 2> destination(extents);
        OutputNoDataPolicy output_no_data_policy(destination.mask(), true);

        fa::convolution::convolve<
            fa::convolve::SkipNoData,
            fa::convolve::DivideByWeights,
            fa::convolve::SkipOutOfImage,
            fa::unary::DiscardRangeErrors>(
                InputNoDataPolicy(source.mask(), true),
                output_no_data_policy,
                fa::sequential, source, kernel_1, destination);

        uint64_t nr_masked_cells;
        fa::statistic::count(fa::sequential, destination.mask(), true,
            nr_masked_cells);
        BOOST_CHECK_EQUAL(nr_masked_cells, nr_rows * nr_cols);
    }

    // Source image with very large values. Convolving these should result
    // in out-оf-range values. It must be possible to detect these and mark
    // them as no-data in the result.
    {
        fern::MaskedArray<double, 2> source(extents);
        std::fill(source.data(), source.data() + source.num_elements(),
            fern::max<double>());
        fern::MaskedArray<double, 2> destination(extents);
        OutputNoDataPolicy output_no_data_policy(destination.mask(), true);

        fa::convolution::convolve<
            fa::convolve::SkipNoData,
            fa::convolve::DivideByWeights,
            fa::convolve::SkipOutOfImage,
            fa::convolve::OutOfRangePolicy>(
                InputNoDataPolicy(source.mask(), true),
                output_no_data_policy,
                fa::sequential, source, kernel_2, destination);

        uint64_t nr_masked_cells;
        fa::statistic::count(fa::sequential, destination.mask(), true,
            nr_masked_cells);
        BOOST_CHECK_EQUAL(nr_masked_cells, nr_rows * nr_cols);
    }
}

BOOST_AUTO_TEST_SUITE_END()
