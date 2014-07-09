#pragma once
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <type_traits>
#include "fern/algorithm/core/index_ranges.h"
#include "fern/algorithm/convolution/neighborhood/square_traits.h"
#include "fern/algorithm/convolution/kernel_traits.h"
#include "fern/core/assert.h"
#include "fern/core/argument_traits.h"
#include "fern/core/base_class.h"
#include "fern/core/thread_client.h"


namespace fern {
namespace convolve {
namespace detail {

template<
    class Value,
    class Result>
struct OutOfRangePolicy
{

    FERN_STATIC_ASSERT(std::is_floating_point, Result)

    inline static bool within_range(
        Result const& result)
    {
        return std::isfinite(result);
    }

};


namespace dispatch {

template<
    bool weigh_values
>
struct ConvolveNorthWestCorner
{
};


template<
    bool weigh_values
>
struct ConvolveNorthEastCorner
{
};


template<
    bool weigh_values
>
struct ConvolveSouthWestCorner
{
};


template<
    bool weigh_values
>
struct ConvolveSouthEastCorner
{
};


template<
    bool weigh_values
>
struct ConvolveNorthSide
{
};


template<
    bool weigh_values
>
struct ConvolveWestSide
{
};


template<
    bool weigh_values
>
struct ConvolveEastSide
{
};


template<
    bool weigh_values
>
struct ConvolveSouthSide
{
};


template<
    bool weigh_values
>
struct ConvolveInnerPart
{
};


template<>
struct ConvolveNorthWestCorner<true>
{

    template<
        class AlternativeForNoDataPolicy,
        class NormalizePolicy,
        class OutOfImagePolicy,
        template<class, class> class OutOfRangePolicy,
        class InputNoDataPolicy,
        class OutputNoDataPolicy,
        class SourceImage,
        class Kernel,
        class DestinationImage>
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SourceImage const& source,
        Kernel const& kernel,
        DestinationImage& destination)
    {
        size_t const radius_{radius(kernel)};
        size_t const first_row_source{0};
        size_t const first_col_source{0};
        size_t first_row_kernel{radius_};
        size_t first_col_kernel;
        size_t nr_rows_kernel{radius_ + 1};
        size_t nr_cols_kernel;
        size_t nr_rows_outside_of_image{radius_};
        size_t nr_cols_outside_of_image;

        value_type<SourceImage> sum_of_values;
        value_type<Kernel> sum_of_weights;
        value_type<SourceImage> out_of_image_value;
        value_type<SourceImage> alternative_value;

        using OORP = OutOfRangePolicy<value_type<SourceImage>,
            value_type<DestinationImage>>;
        using NP = NormalizePolicy;
        using AFNP = AlternativeForNoDataPolicy;

        bool value_seen;

        // Loop over all cells that are situated in the north west corner
        // and for which some cells are outside of the kernel.
        for(size_t row_source = 0; row_source < radius_; ++row_source) {

            first_col_kernel = radius_;
            nr_cols_kernel = radius_ + 1;
            nr_cols_outside_of_image = radius_;

            for(size_t col_source = 0; col_source < radius_; ++col_source) {

                sum_of_values = 0;
                sum_of_weights = 0;
                value_seen = false;

                if(input_no_data_policy.is_no_data(row_source, col_source)) {
                    output_no_data_policy.mark_as_no_data(row_source,
                        col_source);
                }
                else {
                    // We are now positioned on a cell within the image.
                    // When we center the kernel on this cell, part
                    // of the kernel falls outside of the kernel. The
                    // OutOfImagePolicy knows how to calculate values
                    // for these cells.  It returns the value to use or a
                    // signal that no such value exists. This happens when
                    // the policy doesn't want to consider out of image
                    // cells, or when it has to base a new value upon only
                    // no-data values.

                    // Handle cells positioned in the kernel, but outside of
                    // the source image.
                    for(size_t out_of_image_kernel_row = 0;
                            out_of_image_kernel_row < size(kernel, 0);
                            ++out_of_image_kernel_row) {
                        for(size_t out_of_image_kernel_col = 0;
                                out_of_image_kernel_col < size(kernel, 1);
                                    ++out_of_image_kernel_col) {
                            if(out_of_image_kernel_row < first_row_kernel ||
                                out_of_image_kernel_col < first_col_kernel) {

                                if(OutOfImagePolicy::value_north_west(
                                        input_no_data_policy,
                                        source,
                                        out_of_image_kernel_row,
                                        out_of_image_kernel_col,
                                        first_row_kernel, first_col_kernel,
                                        nr_rows_kernel, nr_cols_kernel,
                                        first_row_source, first_col_source,
                                        out_of_image_value)) {
                                    sum_of_values += out_of_image_value *
                                        get(kernel,
                                            out_of_image_kernel_row,
                                            out_of_image_kernel_col);
                                    sum_of_weights += get(kernel,
                                        out_of_image_kernel_row,
                                        out_of_image_kernel_col);
                                    value_seen = true;
                                }
                            }
                        }
                    }

                    // Handle cells positioned in the kernel, and in the source
                    // image.
                    for(size_t row = 0; row < nr_rows_kernel; ++row) {
                        for(size_t col = 0; col < nr_cols_kernel; ++col) {
                            if(input_no_data_policy.is_no_data(
                                    first_row_source + row,
                                    first_col_source + col)) {
                                if(AFNP::value(
                                        input_no_data_policy,
                                        source,
                                        size(source, 0),
                                        size(source, 1),
                                        first_row_source + row,
                                        first_col_source + col,
                                        alternative_value)) {
                                    sum_of_values += alternative_value *
                                        get(kernel, first_row_kernel + row,
                                            first_col_kernel + col);
                                    sum_of_weights +=
                                        get(kernel, first_row_kernel + row,
                                            first_col_kernel + col);
                                    value_seen = true;
                                }
                            }
                            else {
                                sum_of_values +=
                                    get(kernel, first_row_kernel + row,
                                        first_col_kernel + col) *
                                    get(source, first_row_source + row,
                                        first_col_source + col);
                                sum_of_weights +=
                                    get(kernel, first_row_kernel + row,
                                        first_col_kernel + col);
                                value_seen = true;
                            }
                        }
                    }
                }


                // The result can go out of range when the convolution
                // results in an infinite value. Normalizing such a
                // value makes no sense.

                // TODO OutOfRangePolicy must handle integral results too.
                if(!value_seen || !OORP::within_range(sum_of_values)) {
                    output_no_data_policy.mark_as_no_data(row_source,
                        col_source);
                }
                else {
                    get(destination, row_source, col_source) =
                        NP::normalize(sum_of_values, sum_of_weights);
                }

                --first_col_kernel;
                ++nr_cols_kernel;
                --nr_cols_outside_of_image;
            }

            --first_row_kernel;
            ++nr_rows_kernel;
            --nr_rows_outside_of_image;
        }
    }

};


template<>
struct ConvolveNorthEastCorner<true>
{

    template<
        class AlternativeForNoDataPolicy,
        class NormalizePolicy,
        class OutOfImagePolicy,
        template<class, class> class OutOfRangePolicy,
        class InputNoDataPolicy,
        class OutputNoDataPolicy,
        class SourceImage,
        class Kernel,
        class DestinationImage>
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SourceImage const& source,
        Kernel const& kernel,
        DestinationImage& destination)
    {
        size_t const radius_{radius(kernel)};
        size_t const nr_cols_source{size(source, 1)};
        size_t const first_row_source{0};
        size_t first_col_source;
        size_t first_row_kernel{radius_};
        size_t const first_col_kernel{0};
        size_t nr_rows_kernel{radius_ + 1};
        size_t nr_cols_kernel;
        size_t nr_rows_outside_of_image{radius_};
        size_t nr_cols_outside_of_image;

        value_type<SourceImage> sum_of_values;
        value_type<Kernel> sum_of_weights;
        value_type<SourceImage> out_of_image_value;
        value_type<SourceImage> alternative_value;

        using OORP = OutOfRangePolicy<value_type<SourceImage>,
            value_type<DestinationImage>>;
        using NP = NormalizePolicy;
        using AFNP = AlternativeForNoDataPolicy;

        bool value_seen;

        // Loop over all cells that are situated in the north east corner
        // and for which some cells are outside of the kernel.
        for(size_t row_source = 0; row_source < radius_; ++row_source) {

            first_col_source = nr_cols_source - radius_ - radius_;
            nr_cols_kernel = radius_ + radius_;
            nr_cols_outside_of_image = 1;

            for(size_t col_source = nr_cols_source - radius_;
                    col_source < nr_cols_source; ++col_source) {

                sum_of_values = 0;
                sum_of_weights = 0;
                value_seen = false;

                if(input_no_data_policy.is_no_data(row_source, col_source)) {
                    output_no_data_policy.mark_as_no_data(row_source,
                        col_source);
                }
                else {
                    // Handle cells positioned in the kernel, but outside of
                    // the source image.
                    for(size_t out_of_image_kernel_row = 0;
                            out_of_image_kernel_row < size(kernel, 0);
                            ++out_of_image_kernel_row) {
                        for(size_t out_of_image_kernel_col = 0;
                                out_of_image_kernel_col < size(kernel, 1);
                                    ++out_of_image_kernel_col) {
                            if(out_of_image_kernel_row < first_row_kernel ||
                                out_of_image_kernel_col >= nr_cols_kernel) {

                                if(OutOfImagePolicy::value_north_east(
                                        input_no_data_policy,
                                        source,
                                        out_of_image_kernel_row,
                                        out_of_image_kernel_col,
                                        first_row_kernel, first_col_kernel,
                                        nr_rows_kernel, nr_cols_kernel,
                                        first_row_source, first_col_source,
                                        out_of_image_value)) {
                                    sum_of_values += out_of_image_value *
                                        get(kernel,
                                            out_of_image_kernel_row,
                                            out_of_image_kernel_col);
                                    sum_of_weights += get(kernel,
                                        out_of_image_kernel_row,
                                        out_of_image_kernel_col);
                                    value_seen = true;
                                }
                            }
                        }
                    }

                    // Handle cells positioned in the kernel, and in the source
                    // image.
                    for(size_t row = 0; row < nr_rows_kernel; ++row) {
                        for(size_t col = 0; col < nr_cols_kernel; ++col) {
                            if(input_no_data_policy.is_no_data(
                                    first_row_source + row,
                                    first_col_source + col)) {
                                if(AFNP::value(
                                        input_no_data_policy,
                                        source,
                                        size(source, 0),
                                        size(source, 1),
                                        first_row_source + row,
                                        first_col_source + col,
                                        alternative_value)) {
                                    sum_of_values += alternative_value *
                                        get(kernel, first_row_kernel + row,
                                            first_col_kernel + col);
                                    sum_of_weights +=
                                        get(kernel, first_row_kernel + row,
                                            first_col_kernel + col);
                                    value_seen = true;
                                }
                            }
                            else {
                                sum_of_values +=
                                    get(kernel, first_row_kernel + row,
                                        first_col_kernel + col) *
                                    get(source, first_row_source + row,
                                        first_col_source + col);
                                sum_of_weights +=
                                    get(kernel, first_row_kernel + row,
                                        first_col_kernel + col);
                                value_seen = true;
                            }
                        }
                    }
                }


                // The result can go out of range when the convolution
                // results in an infinite value. Normalizing such a
                // value makes no sense.

                // TODO OutOfRangePolicy must handle integral results too.
                if(!value_seen || !OORP::within_range(sum_of_values)) {
                    output_no_data_policy.mark_as_no_data(row_source,
                        col_source);
                }
                else {
                    get(destination, row_source, col_source) =
                        NP::normalize(sum_of_values, sum_of_weights);
                }

                ++first_col_source;
                --nr_cols_kernel;
                ++nr_cols_outside_of_image;
            }

            --first_row_kernel;
            ++nr_rows_kernel;
            --nr_rows_outside_of_image;
        }
    }

};


template<>
struct ConvolveSouthWestCorner<true>
{

    template<
        class AlternativeForNoDataPolicy,
        class NormalizePolicy,
        class OutOfImagePolicy,
        template<class, class> class OutOfRangePolicy,
        class InputNoDataPolicy,
        class OutputNoDataPolicy,
        class SourceImage,
        class Kernel,
        class DestinationImage>
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SourceImage const& source,
        Kernel const& kernel,
        DestinationImage& destination)
    {
        size_t const radius_{radius(kernel)};
        size_t const nr_rows_source{size(source, 0)};
        size_t first_row_source = nr_rows_source - radius_ - radius_;
        size_t const first_col_source{0};
        size_t const first_row_kernel{0};
        size_t first_col_kernel;
        size_t nr_rows_kernel = radius_ + radius_;
        size_t nr_cols_kernel;
        size_t nr_rows_outside_of_image;
        size_t nr_cols_outside_of_image;

        value_type<SourceImage> sum_of_values;
        value_type<Kernel> sum_of_weights;
        value_type<SourceImage> out_of_image_value;
        value_type<SourceImage> alternative_value;

        using OORP = OutOfRangePolicy<value_type<SourceImage>,
            value_type<DestinationImage>>;
        using NP = NormalizePolicy;
        using AFNP = AlternativeForNoDataPolicy;

        bool value_seen;

        // Loop over all cells that are situated in the south west corner
        // and for which some cells are outside of the kernel.
        for(size_t row_source = nr_rows_source - radius_;
                row_source < nr_rows_source; ++row_source) {

            first_col_kernel = radius_;
            nr_cols_kernel = radius_ + 1;
            nr_cols_outside_of_image = radius_;

            for(size_t col_source = 0; col_source < radius_; ++col_source) {

                sum_of_values = 0;
                sum_of_weights = 0;
                value_seen = false;

                if(input_no_data_policy.is_no_data(row_source, col_source)) {
                    output_no_data_policy.mark_as_no_data(row_source,
                        col_source);
                }
                else {
                    // Handle cells positioned in the kernel, but outside of
                    // the source image.
                    for(size_t out_of_image_kernel_row = 0;
                            out_of_image_kernel_row < size(kernel, 0);
                            ++out_of_image_kernel_row) {
                        for(size_t out_of_image_kernel_col = 0;
                                out_of_image_kernel_col < size(kernel, 1);
                                    ++out_of_image_kernel_col) {
                            if(out_of_image_kernel_row >= nr_rows_kernel ||
                                out_of_image_kernel_col < first_col_kernel) {

                                if(OutOfImagePolicy::value_south_west(
                                        input_no_data_policy,
                                        source,
                                        out_of_image_kernel_row,
                                        out_of_image_kernel_col,
                                        first_row_kernel, first_col_kernel,
                                        nr_rows_kernel, nr_cols_kernel,
                                        first_row_source, first_col_source,
                                        out_of_image_value)) {
                                    sum_of_values += out_of_image_value *
                                        get(kernel,
                                            out_of_image_kernel_row,
                                            out_of_image_kernel_col);
                                    sum_of_weights += get(kernel,
                                        out_of_image_kernel_row,
                                        out_of_image_kernel_col);
                                    value_seen = true;
                                }
                            }
                        }
                    }

                    // Handle cells positioned in the kernel, and in the source
                    // image.
                    for(size_t row = 0; row < nr_rows_kernel; ++row) {
                        for(size_t col = 0; col < nr_cols_kernel; ++col) {
                            if(input_no_data_policy.is_no_data(
                                    first_row_source + row,
                                    first_col_source + col)) {
                                if(AFNP::value(
                                        input_no_data_policy,
                                        source,
                                        size(source, 0),
                                        size(source, 1),
                                        first_row_source + row,
                                        first_col_source + col,
                                        alternative_value)) {
                                    sum_of_values += alternative_value *
                                        get(kernel, first_row_kernel + row,
                                            first_col_kernel + col);
                                    sum_of_weights +=
                                        get(kernel, first_row_kernel + row,
                                            first_col_kernel + col);
                                    value_seen = true;
                                }
                            }
                            else {
                                sum_of_values +=
                                    get(kernel, first_row_kernel + row,
                                        first_col_kernel + col) *
                                    get(source, first_row_source + row,
                                        first_col_source + col);
                                sum_of_weights +=
                                    get(kernel, first_row_kernel + row,
                                        first_col_kernel + col);
                                value_seen = true;
                            }
                        }
                    }
                }


                // The result can go out of range when the convolution
                // results in an infinite value. Normalizing such a
                // value makes no sense.

                // TODO OutOfRangePolicy must handle integral results too.
                if(!value_seen || !OORP::within_range(sum_of_values)) {
                    output_no_data_policy.mark_as_no_data(row_source,
                        col_source);
                }
                else {
                    get(destination, row_source, col_source) =
                        NP::normalize(sum_of_values, sum_of_weights);
                }

                --first_col_kernel;
                ++nr_cols_kernel;
                --nr_cols_outside_of_image;
            }

            ++first_row_source;
            --nr_rows_kernel;
            ++nr_rows_outside_of_image;
        }
    }

};


template<>
struct ConvolveSouthEastCorner<true>
{

    template<
        class AlternativeForNoDataPolicy,
        class NormalizePolicy,
        class OutOfImagePolicy,
        template<class, class> class OutOfRangePolicy,
        class InputNoDataPolicy,
        class OutputNoDataPolicy,
        class SourceImage,
        class Kernel,
        class DestinationImage>
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SourceImage const& source,
        Kernel const& kernel,
        DestinationImage& destination)
    {
        size_t const radius_{radius(kernel)};
        size_t const nr_rows_source{size(source, 0)};
        size_t const nr_cols_source{size(source, 1)};
        size_t first_row_source = nr_rows_source - radius_ - radius_;
        size_t first_col_source;
        size_t const first_row_kernel{0};
        size_t const first_col_kernel{0};
        size_t nr_rows_kernel = radius_ + radius_;
        size_t nr_cols_kernel;
        size_t nr_rows_outside_of_image;
        size_t nr_cols_outside_of_image;

        value_type<SourceImage> sum_of_values;
        value_type<Kernel> sum_of_weights;
        value_type<SourceImage> out_of_image_value;
        value_type<SourceImage> alternative_value;

        using OORP = OutOfRangePolicy<value_type<SourceImage>,
            value_type<DestinationImage>>;
        using NP = NormalizePolicy;
        using AFNP = AlternativeForNoDataPolicy;

        bool value_seen;

        // Loop over all cells that are situated in the south west corner
        // and for which some cells are outside of the kernel.
        for(size_t row_source = nr_rows_source - radius_;
                row_source < nr_rows_source; ++row_source) {

            first_col_source = nr_cols_source - radius_ - radius_;
            nr_cols_kernel = radius_ + radius_;
            nr_cols_outside_of_image = 1;

            for(size_t col_source = nr_cols_source - radius_;
                    col_source < nr_cols_source; ++col_source) {

                sum_of_values = 0;
                sum_of_weights = 0;
                value_seen = false;

                if(input_no_data_policy.is_no_data(row_source, col_source)) {
                    output_no_data_policy.mark_as_no_data(row_source,
                        col_source);
                }
                else {
                    // Handle cells positioned in the kernel, but outside of
                    // the source image.
                    for(size_t out_of_image_kernel_row = 0;
                            out_of_image_kernel_row < size(kernel, 0);
                            ++out_of_image_kernel_row) {
                        for(size_t out_of_image_kernel_col = 0;
                                out_of_image_kernel_col < size(kernel, 1);
                                    ++out_of_image_kernel_col) {

                            if(out_of_image_kernel_row >= nr_rows_kernel ||
                                out_of_image_kernel_col >= nr_cols_kernel) {

                                if(OutOfImagePolicy::value_south_east(
                                        input_no_data_policy,
                                        source,
                                        out_of_image_kernel_row,
                                        out_of_image_kernel_col,
                                        first_row_kernel, first_col_kernel,
                                        nr_rows_kernel, nr_cols_kernel,
                                        first_row_source, first_col_source,
                                        out_of_image_value)) {
                                    sum_of_values += out_of_image_value *
                                        get(kernel,
                                            out_of_image_kernel_row,
                                            out_of_image_kernel_col);
                                    sum_of_weights += get(kernel,
                                        out_of_image_kernel_row,
                                        out_of_image_kernel_col);
                                    value_seen = true;
                                }
                            }
                        }
                    }

                    // Handle cells positioned in the kernel, and in the source
                    // image.
                    for(size_t row = 0; row < nr_rows_kernel; ++row) {
                        for(size_t col = 0; col < nr_cols_kernel; ++col) {
                            if(input_no_data_policy.is_no_data(
                                    first_row_source + row,
                                    first_col_source + col)) {
                                if(AFNP::value(
                                        input_no_data_policy,
                                        source,
                                        size(source, 0),
                                        size(source, 1),
                                        first_row_source + row,
                                        first_col_source + col,
                                        alternative_value)) {
                                    sum_of_values += alternative_value *
                                        get(kernel, first_row_kernel + row,
                                            first_col_kernel + col);
                                    sum_of_weights +=
                                        get(kernel, first_row_kernel + row,
                                            first_col_kernel + col);
                                    value_seen = true;
                                }
                            }
                            else {
                                sum_of_values +=
                                    get(kernel, first_row_kernel + row,
                                        first_col_kernel + col) *
                                    get(source, first_row_source + row,
                                        first_col_source + col);
                                sum_of_weights +=
                                    get(kernel, first_row_kernel + row,
                                        first_col_kernel + col);
                                value_seen = true;
                            }
                        }
                    }
                }


                // The result can go out of range when the convolution
                // results in an infinite value. Normalizing such a
                // value makes no sense.

                // TODO OutOfRangePolicy must handle integral results too.
                if(!value_seen || !OORP::within_range(sum_of_values)) {
                    output_no_data_policy.mark_as_no_data(row_source,
                        col_source);
                }
                else {
                    get(destination, row_source, col_source) =
                        NP::normalize(sum_of_values, sum_of_weights);
                }

                ++first_col_source;
                --nr_cols_kernel;
                ++nr_cols_outside_of_image;
            }

            ++first_row_source;
            --nr_rows_kernel;
            ++nr_rows_outside_of_image;
        }
    }

};


template<>
struct ConvolveNorthSide<true>
{

    template<
        class AlternativeForNoDataPolicy,
        class NormalizePolicy,
        class OutOfImagePolicy,
        template<class, class> class OutOfRangePolicy,
        class InputNoDataPolicy,
        class OutputNoDataPolicy,
        class SourceImage,
        class Kernel,
        class DestinationImage>
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SourceImage const& source,
        Kernel const& kernel,
        DestinationImage& destination)
    {
        size_t const radius_{radius(kernel)};
        size_t const nr_cols_source{size(source, 1)};
        size_t const first_row_source{0};
        size_t first_col_source;
        size_t first_row_kernel{radius_};
        size_t const first_col_kernel{0};
        size_t nr_rows_kernel{radius_ + 1};
        size_t const nr_cols_kernel{width(kernel)};
        size_t nr_rows_outside_of_image{radius_};

        value_type<SourceImage> sum_of_values;
        value_type<Kernel> sum_of_weights;
        value_type<SourceImage> out_of_image_value;
        value_type<SourceImage> alternative_value;

        using OORP = OutOfRangePolicy<value_type<SourceImage>,
            value_type<DestinationImage>>;
        using NP = NormalizePolicy;
        using AFNP = AlternativeForNoDataPolicy;

        bool value_seen;

        // Loop over all cells that are situated in the north side and for
        // which some cells are outside of the kernel.
        for(size_t row_source = 0; row_source < radius_; ++row_source) {

            first_col_source = 0;

            for(size_t col_source = radius_; col_source <
                    nr_cols_source - radius_; ++col_source) {

                sum_of_values = 0;
                sum_of_weights = 0;
                value_seen = false;

                if(input_no_data_policy.is_no_data(row_source, col_source)) {
                    output_no_data_policy.mark_as_no_data(row_source,
                        col_source);
                }
                else {
                    // Handle cells positioned in the kernel, but outside of
                    // the source image.
                    for(size_t out_of_image_kernel_row = 0;
                            out_of_image_kernel_row < size(kernel, 0);
                            ++out_of_image_kernel_row) {
                        for(size_t out_of_image_kernel_col = 0;
                                out_of_image_kernel_col < size(kernel, 1);
                                ++out_of_image_kernel_col) {

                            assert(out_of_image_kernel_col >= first_col_kernel);

                            if(out_of_image_kernel_row < first_row_kernel) {

                                if(OutOfImagePolicy::value_north(
                                        input_no_data_policy,
                                        source,
                                        out_of_image_kernel_row,
                                        out_of_image_kernel_col,
                                        first_row_kernel, first_col_kernel,
                                        nr_rows_kernel, nr_cols_kernel,
                                        first_row_source, first_col_source,
                                        out_of_image_value)) {
                                    sum_of_values += out_of_image_value *
                                        get(kernel,
                                            out_of_image_kernel_row,
                                            out_of_image_kernel_col);
                                    sum_of_weights += get(kernel,
                                        out_of_image_kernel_row,
                                        out_of_image_kernel_col);
                                    value_seen = true;
                                }
                            }
                        }
                    }

                    // Handle cells positioned in the kernel, and in the source
                    // image.
                    for(size_t row = 0; row < nr_rows_kernel; ++row) {
                        for(size_t col = 0; col < nr_cols_kernel; ++col) {
                            if(input_no_data_policy.is_no_data(
                                    first_row_source + row,
                                    first_col_source + col)) {
                                if(AFNP::value(
                                        input_no_data_policy,
                                        source,
                                        size(source, 0),
                                        size(source, 1),
                                        first_row_source + row,
                                        first_col_source + col,
                                        alternative_value)) {
                                    sum_of_values += alternative_value *
                                        get(kernel, first_row_kernel + row,
                                            first_col_kernel + col);
                                    sum_of_weights +=
                                        get(kernel, first_row_kernel + row,
                                            first_col_kernel + col);
                                    value_seen = true;
                                }
                            }
                            else {
                                sum_of_values +=
                                    get(kernel, first_row_kernel + row,
                                        first_col_kernel + col) *
                                    get(source, first_row_source + row,
                                        first_col_source + col);
                                sum_of_weights +=
                                    get(kernel, first_row_kernel + row,
                                        first_col_kernel + col);
                                value_seen = true;
                            }
                        }
                    }
                }


                // The result can go out of range when the convolution
                // results in an infinite value. Normalizing such a
                // value makes no sense.

                // TODO OutOfRangePolicy must handle integral results too.
                if(!value_seen || !OORP::within_range(sum_of_values)) {
                    output_no_data_policy.mark_as_no_data(row_source,
                        col_source);
                }
                else {
                    get(destination, row_source, col_source) =
                        NP::normalize(sum_of_values, sum_of_weights);
                }

                ++first_col_source;
            }

            --first_row_kernel;
            ++nr_rows_kernel;
            --nr_rows_outside_of_image;
        }
    }

};


template<>
struct ConvolveWestSide<true>
{

    template<
        class AlternativeForNoDataPolicy,
        class NormalizePolicy,
        class OutOfImagePolicy,
        template<class, class> class OutOfRangePolicy,
        class InputNoDataPolicy,
        class OutputNoDataPolicy,
        class SourceImage,
        class Kernel,
        class DestinationImage>
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SourceImage const& source,
        Kernel const& kernel,
        DestinationImage& destination)
    {
        size_t const radius_{radius(kernel)};
        size_t const nr_rows_source{size(source, 0)};
        size_t first_row_source{0};
        size_t const first_col_source{0};
        size_t const first_row_kernel{0};
        size_t first_col_kernel;
        size_t const nr_rows_kernel{height(kernel)};
        size_t nr_cols_kernel;
        size_t nr_cols_outside_of_image;

        value_type<SourceImage> sum_of_values;
        value_type<Kernel> sum_of_weights;
        value_type<SourceImage> out_of_image_value;
        value_type<SourceImage> alternative_value;

        using OORP = OutOfRangePolicy<value_type<SourceImage>,
            value_type<DestinationImage>>;
        using NP = NormalizePolicy;
        using AFNP = AlternativeForNoDataPolicy;

        bool value_seen;

        // Loop over all cells that are situated in the west side and for
        // which some cells are outside of the kernel.
        for(size_t row_source = radius_; row_source <
                nr_rows_source - radius_; ++row_source) {

            first_col_kernel = radius_;
            nr_cols_kernel = radius_ + 1;
            nr_cols_outside_of_image = radius_;

            for(size_t col_source = 0; col_source < radius_; ++col_source) {

                sum_of_values = 0;
                sum_of_weights = 0;
                value_seen = false;

                if(input_no_data_policy.is_no_data(row_source, col_source)) {
                    output_no_data_policy.mark_as_no_data(row_source,
                        col_source);
                }
                else {
                    // Handle cells positioned in the kernel, but outside of
                    // the source image.
                    for(size_t out_of_image_kernel_row = 0;
                            out_of_image_kernel_row < size(kernel, 0);
                            ++out_of_image_kernel_row) {
                        for(size_t out_of_image_kernel_col = 0;
                                out_of_image_kernel_col < size(kernel, 1);
                                ++out_of_image_kernel_col) {

                            assert(out_of_image_kernel_row >= first_row_kernel);

                            if(out_of_image_kernel_col < first_col_kernel) {

                                if(OutOfImagePolicy::value_west(
                                        input_no_data_policy,
                                        source,
                                        out_of_image_kernel_row,
                                        out_of_image_kernel_col,
                                        first_row_kernel, first_col_kernel,
                                        nr_rows_kernel, nr_cols_kernel,
                                        first_row_source, first_col_source,
                                        out_of_image_value)) {
                                    sum_of_values += out_of_image_value *
                                        get(kernel,
                                            out_of_image_kernel_row,
                                            out_of_image_kernel_col);
                                    sum_of_weights += get(kernel,
                                        out_of_image_kernel_row,
                                        out_of_image_kernel_col);
                                    value_seen = true;
                                }
                            }
                        }
                    }

                    // Handle cells positioned in the kernel, and in the source
                    // image.
                    for(size_t row = 0; row < nr_rows_kernel; ++row) {
                        for(size_t col = 0; col < nr_cols_kernel; ++col) {
                            if(input_no_data_policy.is_no_data(
                                    first_row_source + row,
                                    first_col_source + col)) {
                                if(AFNP::value(
                                        input_no_data_policy,
                                        source,
                                        size(source, 0),
                                        size(source, 1),
                                        first_row_source + row,
                                        first_col_source + col,
                                        alternative_value)) {
                                    sum_of_values += alternative_value *
                                        get(kernel, first_row_kernel + row,
                                            first_col_kernel + col);
                                    sum_of_weights +=
                                        get(kernel, first_row_kernel + row,
                                            first_col_kernel + col);
                                    value_seen = true;
                                }
                            }
                            else {
                                sum_of_values +=
                                    get(kernel, first_row_kernel + row,
                                        first_col_kernel + col) *
                                    get(source, first_row_source + row,
                                        first_col_source + col);
                                sum_of_weights +=
                                    get(kernel, first_row_kernel + row,
                                        first_col_kernel + col);
                                value_seen = true;
                            }
                        }
                    }
                }


                // The result can go out of range when the convolution
                // results in an infinite value. Normalizing such a
                // value makes no sense.

                // TODO OutOfRangePolicy must handle integral results too.
                if(!value_seen || !OORP::within_range(sum_of_values)) {
                    output_no_data_policy.mark_as_no_data(row_source,
                        col_source);
                }
                else {
                    get(destination, row_source, col_source) =
                        NP::normalize(sum_of_values, sum_of_weights);
                }

                --first_col_kernel;
                ++nr_cols_kernel;
                --nr_cols_outside_of_image;
            }

            ++first_row_source;
        }
    }

};


template<>
struct ConvolveEastSide<true>
{

    template<
        class AlternativeForNoDataPolicy,
        class NormalizePolicy,
        class OutOfImagePolicy,
        template<class, class> class OutOfRangePolicy,
        class InputNoDataPolicy,
        class OutputNoDataPolicy,
        class SourceImage,
        class Kernel,
        class DestinationImage>
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SourceImage const& source,
        Kernel const& kernel,
        DestinationImage& destination)
    {
        size_t const radius_{radius(kernel)};
        size_t const nr_rows_source{size(source, 0)};
        size_t const nr_cols_source{size(source, 1)};
        size_t first_row_source{0};
        size_t first_col_source;
        size_t const first_row_kernel{0};
        size_t const first_col_kernel{0};
        size_t const nr_rows_kernel{height(kernel)};
        size_t nr_cols_kernel;
        size_t nr_cols_outside_of_image;

        value_type<SourceImage> sum_of_values;
        value_type<Kernel> sum_of_weights;
        value_type<SourceImage> out_of_image_value;
        value_type<SourceImage> alternative_value;

        using OORP = OutOfRangePolicy<value_type<SourceImage>,
            value_type<DestinationImage>>;
        using NP = NormalizePolicy;
        using AFNP = AlternativeForNoDataPolicy;

        bool value_seen;

        // Loop over all cells that are situated in the east side and for
        // which some cells are outside of the kernel.
        for(size_t row_source = radius_; row_source <
                nr_rows_source - radius_; ++row_source) {

            first_col_source = nr_cols_source - radius_ - radius_;
            nr_cols_kernel = radius_ + radius_;
            nr_cols_outside_of_image = 1;

            for(size_t col_source = nr_cols_source - radius_;
                    col_source < nr_cols_source; ++col_source) {

                sum_of_values = 0;
                sum_of_weights = 0;
                value_seen = false;

                if(input_no_data_policy.is_no_data(row_source, col_source)) {
                    output_no_data_policy.mark_as_no_data(row_source,
                        col_source);
                }
                else {
                    // Handle cells positioned in the kernel, but outside of
                    // the source image.
                    for(size_t out_of_image_kernel_row = 0;
                            out_of_image_kernel_row < size(kernel, 0);
                            ++out_of_image_kernel_row) {
                        for(size_t out_of_image_kernel_col = 0;
                                out_of_image_kernel_col < size(kernel, 1);
                                ++out_of_image_kernel_col) {

                            assert(out_of_image_kernel_row >= first_row_kernel);

                            if(out_of_image_kernel_col >= nr_cols_kernel) {

                                if(OutOfImagePolicy::value_east(
                                        input_no_data_policy,
                                        source,
                                        out_of_image_kernel_row,
                                        out_of_image_kernel_col,
                                        first_row_kernel, first_col_kernel,
                                        nr_rows_kernel, nr_cols_kernel,
                                        first_row_source, first_col_source,
                                        out_of_image_value)) {
                                    sum_of_values += out_of_image_value *
                                        get(kernel,
                                            out_of_image_kernel_row,
                                            out_of_image_kernel_col);
                                    sum_of_weights += get(kernel,
                                        out_of_image_kernel_row,
                                        out_of_image_kernel_col);
                                    value_seen = true;
                                }
                            }
                        }
                    }

                    // Handle cells positioned in the kernel, and in the source
                    // image.
                    for(size_t row = 0; row < nr_rows_kernel; ++row) {
                        for(size_t col = 0; col < nr_cols_kernel; ++col) {
                            if(input_no_data_policy.is_no_data(
                                    first_row_source + row,
                                    first_col_source + col)) {
                                if(AFNP::value(
                                        input_no_data_policy,
                                        source,
                                        size(source, 0),
                                        size(source, 1),
                                        first_row_source + row,
                                        first_col_source + col,
                                        alternative_value)) {
                                    sum_of_values += alternative_value *
                                        get(kernel, first_row_kernel + row,
                                            first_col_kernel + col);
                                    sum_of_weights +=
                                        get(kernel, first_row_kernel + row,
                                            first_col_kernel + col);
                                    value_seen = true;
                                }
                            }
                            else {
                                sum_of_values +=
                                    get(kernel, first_row_kernel + row,
                                        first_col_kernel + col) *
                                    get(source, first_row_source + row,
                                        first_col_source + col);
                                sum_of_weights +=
                                    get(kernel, first_row_kernel + row,
                                        first_col_kernel + col);
                                value_seen = true;
                            }
                        }
                    }
                }


                // The result can go out of range when the convolution
                // results in an infinite value. Normalizing such a
                // value makes no sense.

                // TODO OutOfRangePolicy must handle integral results too.
                if(!value_seen || !OORP::within_range(sum_of_values)) {
                    output_no_data_policy.mark_as_no_data(row_source,
                        col_source);
                }
                else {
                    get(destination, row_source, col_source) =
                        NP::normalize(sum_of_values, sum_of_weights);
                }

                ++first_col_source;
                --nr_cols_kernel;
                ++nr_cols_outside_of_image;
            }

            ++first_row_source;
        }
    }

};


template<>
struct ConvolveSouthSide<true>
{

    template<
        class AlternativeForNoDataPolicy,
        class NormalizePolicy,
        class OutOfImagePolicy,
        template<class, class> class OutOfRangePolicy,
        class InputNoDataPolicy,
        class OutputNoDataPolicy,
        class SourceImage,
        class Kernel,
        class DestinationImage>
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SourceImage const& source,
        Kernel const& kernel,
        DestinationImage& destination)
    {
        size_t const radius_{radius(kernel)};
        size_t const nr_rows_source{size(source, 0)};
        size_t const nr_cols_source{size(source, 1)};
        size_t first_row_source{nr_rows_source - radius_ - radius_};
        size_t first_col_source;
        size_t const first_row_kernel{0};
        size_t const first_col_kernel{0};
        size_t nr_rows_kernel{radius_ + radius_};
        size_t const nr_cols_kernel{width(kernel)};
        size_t nr_rows_outside_of_image{1};

        value_type<SourceImage> sum_of_values;
        value_type<Kernel> sum_of_weights;
        value_type<SourceImage> out_of_image_value;
        value_type<SourceImage> alternative_value;

        using OORP = OutOfRangePolicy<value_type<SourceImage>,
            value_type<DestinationImage>>;
        using NP = NormalizePolicy;
        using AFNP = AlternativeForNoDataPolicy;

        bool value_seen;

        // Loop over all cells that are situated in the south side and for
        // which some cells are outside of the kernel.
        for(size_t row_source = nr_rows_source - radius_;
                row_source < nr_rows_source; ++row_source) {

            first_col_source = 0;

            for(size_t col_source = radius_; col_source <
                    nr_cols_source - radius_; ++col_source) {

                sum_of_values = 0;
                sum_of_weights = 0;
                value_seen = false;

                if(input_no_data_policy.is_no_data(row_source, col_source)) {
                    output_no_data_policy.mark_as_no_data(row_source,
                        col_source);
                }
                else {
                    // Handle cells positioned in the kernel, but outside of
                    // the source image.
                    for(size_t out_of_image_kernel_row = 0;
                            out_of_image_kernel_row < size(kernel, 0);
                            ++out_of_image_kernel_row) {
                        for(size_t out_of_image_kernel_col = 0;
                                out_of_image_kernel_col < size(kernel, 1);
                                ++out_of_image_kernel_col) {

                            assert(out_of_image_kernel_col >= first_col_kernel);

                            if(out_of_image_kernel_row >= nr_rows_kernel) {

                                if(OutOfImagePolicy::value_south(
                                        input_no_data_policy,
                                        source,
                                        out_of_image_kernel_row,
                                        out_of_image_kernel_col,
                                        first_row_kernel, first_col_kernel,
                                        nr_rows_kernel, nr_cols_kernel,
                                        first_row_source, first_col_source,
                                        out_of_image_value)) {
                                    sum_of_values += out_of_image_value *
                                        get(kernel,
                                            out_of_image_kernel_row,
                                            out_of_image_kernel_col);
                                    sum_of_weights += get(kernel,
                                        out_of_image_kernel_row,
                                        out_of_image_kernel_col);
                                    value_seen = true;
                                }
                            }
                        }
                    }

                    // Handle cells positioned in the kernel, and in the source
                    // image.
                    for(size_t row = 0; row < nr_rows_kernel; ++row) {
                        for(size_t col = 0; col < nr_cols_kernel; ++col) {
                            if(input_no_data_policy.is_no_data(
                                    first_row_source + row,
                                    first_col_source + col)) {
                                if(AFNP::value(
                                        input_no_data_policy,
                                        source,
                                        size(source, 0),
                                        size(source, 1),
                                        first_row_source + row,
                                        first_col_source + col,
                                        alternative_value)) {
                                    sum_of_values += alternative_value *
                                        get(kernel, first_row_kernel + row,
                                            first_col_kernel + col);
                                    sum_of_weights +=
                                        get(kernel, first_row_kernel + row,
                                            first_col_kernel + col);
                                    value_seen = true;
                                }
                            }
                            else {
                                sum_of_values +=
                                    get(kernel, first_row_kernel + row,
                                        first_col_kernel + col) *
                                    get(source, first_row_source + row,
                                        first_col_source + col);
                                sum_of_weights +=
                                    get(kernel, first_row_kernel + row,
                                        first_col_kernel + col);
                                value_seen = true;
                            }
                        }
                    }
                }


                // The result can go out of range when the convolution
                // results in an infinite value. Normalizing such a
                // value makes no sense.

                // TODO OutOfRangePolicy must handle integral results too.
                if(!value_seen || !OORP::within_range(sum_of_values)) {
                    output_no_data_policy.mark_as_no_data(row_source,
                        col_source);
                }
                else {
                    get(destination, row_source, col_source) =
                        NP::normalize(sum_of_values, sum_of_weights);
                }

                ++first_col_source;
            }

            // --first_row_kernel;
            ++first_row_source;
            --nr_rows_kernel;
            ++nr_rows_outside_of_image;
        }
    }

};


template<>
struct ConvolveInnerPart<true>
{

    template<
        class AlternativeForNoDataPolicy,
        class NormalizePolicy,
        template<class, class> class OutOfRangePolicy,
        class InputNoDataPolicy,
        class OutputNoDataPolicy,
        class SourceImage,
        class Kernel,
        class DestinationImage>
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        IndexRanges<2> const& index_ranges,
        SourceImage const& source,
        Kernel const& kernel,
        DestinationImage& destination)
    {
        size_t const radius_{radius(kernel)};
        size_t first_row_source{index_ranges[0].begin() - radius_};
        size_t first_col_source;
        size_t const nr_rows_kernel{height(kernel)};
        size_t const nr_cols_kernel{width(kernel)};

        value_type<SourceImage> sum_of_values;
        value_type<Kernel> sum_of_weights;
        value_type<SourceImage> alternative_value;

        using OORP = OutOfRangePolicy<value_type<SourceImage>,
            value_type<DestinationImage>>;
        using NP = NormalizePolicy;
        using AFNP = AlternativeForNoDataPolicy;

        bool value_seen;

        // Loop over all cells that are situated in the inner part. The kernel
        // does not extent outside of the source image.
        for(size_t row_source = index_ranges[0].begin(); row_source <
                index_ranges[0].end(); ++row_source) {

            first_col_source = index_ranges[1].begin() - radius_;

            for(size_t col_source = index_ranges[1].begin(); col_source <
                    index_ranges[1].end(); ++col_source) {

                sum_of_values = 0;
                sum_of_weights = 0;
                value_seen = false;

                if(input_no_data_policy.is_no_data(row_source, col_source)) {
                    output_no_data_policy.mark_as_no_data(row_source,
                        col_source);
                }
                else {
                    // Handle cells positioned in the kernel, and in the source
                    // image.
                    for(size_t row = 0; row < nr_rows_kernel; ++row) {
                        for(size_t col = 0; col < nr_cols_kernel; ++col) {
                            if(input_no_data_policy.is_no_data(
                                    first_row_source + row,
                                    first_col_source + col)) {
                                if(AFNP::value(
                                        input_no_data_policy,
                                        source,
                                        fern::size(source, 0),
                                        fern::size(source, 1),
                                        first_row_source + row,
                                        first_col_source + col,
                                        alternative_value)) {
                                    sum_of_values += alternative_value *
                                        get(kernel, row, col);
                                    sum_of_weights += get(kernel, row, col);
                                    value_seen = true;
                                }
                            }
                            else {
                                sum_of_values +=
                                    get(kernel, row, col) *
                                    get(source, first_row_source + row,
                                        first_col_source + col);
                                sum_of_weights +=
                                    get(kernel, row, col);
                                value_seen = true;
                            }
                        }
                    }
                }


                // The result can go out of range when the convolution
                // results in an infinite value. Normalizing such a
                // value makes no sense.

                // TODO OutOfRangePolicy must handle integral results too.
                if(!value_seen || !OORP::within_range(sum_of_values)) {
                    output_no_data_policy.mark_as_no_data(row_source,
                        col_source);
                }
                else {
                    get(destination, row_source, col_source) =
                        NP::normalize(sum_of_values, sum_of_weights);
                }

                ++first_col_source;
            }

            ++first_row_source;
        }
    }

};


template<
    class AlternativeForNoDataPolicy,
    class NormalizePolicy,
    class OutOfImagePolicy,
    template<class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class SourceImage,
    class Kernel,
    class DestinationImage,
    class ExecutionPolicy>
struct Convolve
{
};


template<
    class AlternativeForNoDataPolicy,
    class NormalizePolicy,
    class OutOfImagePolicy,
    template<class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class SourceImage,
    class Kernel,
    class DestinationImage>
struct Convolve<
    AlternativeForNoDataPolicy,
    NormalizePolicy,
    OutOfImagePolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    SourceImage,
    Kernel,
    DestinationImage,
    SequentialExecutionPolicy>
{
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy const& /* execution_policy */,
        SourceImage const& source,
        Kernel const& kernel,
        DestinationImage& destination)
    {
        // Corners.
        dispatch::ConvolveNorthWestCorner<KernelTraits<Kernel>::weigh_values>::
            template apply<AlternativeForNoDataPolicy, NormalizePolicy,
                OutOfImagePolicy, OutOfRangePolicy>(
                    input_no_data_policy, output_no_data_policy,
                    source, kernel, destination);
        dispatch::ConvolveNorthEastCorner<KernelTraits<Kernel>::weigh_values>::
            template apply<AlternativeForNoDataPolicy, NormalizePolicy,
                OutOfImagePolicy, OutOfRangePolicy>(
                    input_no_data_policy, output_no_data_policy,
                    source, kernel, destination);
        dispatch::ConvolveSouthWestCorner<KernelTraits<Kernel>::weigh_values>::
            template apply<AlternativeForNoDataPolicy, NormalizePolicy,
                OutOfImagePolicy, OutOfRangePolicy>(
                    input_no_data_policy, output_no_data_policy,
                    source, kernel, destination);
        dispatch::ConvolveSouthEastCorner<KernelTraits<Kernel>::weigh_values>::
            template apply<AlternativeForNoDataPolicy, NormalizePolicy,
                OutOfImagePolicy, OutOfRangePolicy>(
                    input_no_data_policy, output_no_data_policy,
                    source, kernel, destination);

        // Sides.
        dispatch::ConvolveNorthSide<KernelTraits<Kernel>::weigh_values>::
            template apply<AlternativeForNoDataPolicy, NormalizePolicy,
                OutOfImagePolicy, OutOfRangePolicy>(
                    input_no_data_policy, output_no_data_policy,
                    source, kernel, destination);
        dispatch::ConvolveWestSide<KernelTraits<Kernel>::weigh_values>::
            template apply<AlternativeForNoDataPolicy, NormalizePolicy,
                OutOfImagePolicy, OutOfRangePolicy>(
                    input_no_data_policy, output_no_data_policy,
                    source, kernel, destination);
        dispatch::ConvolveEastSide<KernelTraits<Kernel>::weigh_values>::
            template apply<AlternativeForNoDataPolicy, NormalizePolicy,
                OutOfImagePolicy, OutOfRangePolicy>(
                    input_no_data_policy, output_no_data_policy,
                    source, kernel, destination);
        dispatch::ConvolveSouthSide<KernelTraits<Kernel>::weigh_values>::
            template apply<AlternativeForNoDataPolicy, NormalizePolicy,
                OutOfImagePolicy, OutOfRangePolicy>(
                    input_no_data_policy, output_no_data_policy,
                    source, kernel, destination);

        // Inner part.
        dispatch::ConvolveInnerPart<KernelTraits<Kernel>::weigh_values>::
            template apply<AlternativeForNoDataPolicy, NormalizePolicy,
                OutOfRangePolicy>(
                    input_no_data_policy, output_no_data_policy,
                    IndexRanges<2>{
                        IndexRange(fern::radius(kernel),
                            fern::size(source, 0) - fern::radius(kernel)),
                        IndexRange(fern::radius(kernel),
                            fern::size(source, 1) - fern::radius(kernel))},
                    source, kernel, destination);
    }

};


#define CREATE_BORDER_TASK(                                                    \
    part)                                                                      \
{                                                                              \
    auto function = std::bind((BorderFunction)                                 \
        dispatch::Convolve##part<                                              \
            KernelTraits<Kernel>::weigh_values>::template                      \
                apply<AlternativeForNoDataPolicy, NormalizePolicy,             \
                    OutOfImagePolicy, OutOfRangePolicy>,                       \
        std::cref(input_no_data_policy),                                       \
        std::ref(output_no_data_policy),                                       \
        std::cref(source), std::cref(kernel), std::ref(destination));          \
    futures.emplace_back(pool.submit(function));                               \
}

#define CREATE_INNER_PART_TASK(                                                \
    index_ranges)                                                              \
{                                                                              \
    auto function = std::bind((InnerPartFunction)                              \
        dispatch::ConvolveInnerPart<                                           \
            KernelTraits<Kernel>::weigh_values>::template                      \
                apply<AlternativeForNoDataPolicy, NormalizePolicy,             \
                    OutOfRangePolicy>,                                         \
        std::cref(input_no_data_policy),                                       \
        std::ref(output_no_data_policy),                                       \
        std::cref(index_ranges),                                               \
        std::cref(source), std::cref(kernel), std::ref(destination));          \
    futures.emplace_back(pool.submit(function));                               \
}


template<
    class AlternativeForNoDataPolicy,
    class NormalizePolicy,
    class OutOfImagePolicy,
    template<class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class SourceImage,
    class Kernel,
    class DestinationImage>
struct Convolve<
    AlternativeForNoDataPolicy,
    NormalizePolicy,
    OutOfImagePolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    SourceImage,
    Kernel,
    DestinationImage,
    ParallelExecutionPolicy>
{
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy const& /* execution_policy */,
        SourceImage const& source,
        Kernel const& kernel,
        DestinationImage& destination)
    {
        ThreadPool& pool(ThreadClient::pool());
        std::vector<std::future<void>> futures;

        using BorderFunction = void(*)(InputNoDataPolicy const&,
            OutputNoDataPolicy&, SourceImage const&, Kernel const&,
            DestinationImage&);
        using InnerPartFunction = void(*)(InputNoDataPolicy const&,
            OutputNoDataPolicy&, IndexRanges<2> const&, SourceImage const&,
            Kernel const&, DestinationImage&);

        // Corners.
        CREATE_BORDER_TASK(NorthWestCorner)
        CREATE_BORDER_TASK(NorthEastCorner)
        CREATE_BORDER_TASK(SouthWestCorner)
        CREATE_BORDER_TASK(SouthEastCorner)

        // Sides.
        CREATE_BORDER_TASK(NorthSide)
        CREATE_BORDER_TASK(WestSide)
        CREATE_BORDER_TASK(EastSide)
        CREATE_BORDER_TASK(SouthSide)

        // Inner part.
        // Divide the inner part in a number of pieces that can be handled
        // concurrently.
        assert(fern::size(source, 0) == fern::size(destination, 0));
        assert(fern::size(source, 1) == fern::size(destination, 1));
        assert(2 * fern::radius(kernel) <= fern::size(source, 0));
        assert(2 * fern::radius(kernel) <= fern::size(source, 1));

        size_t const size1 = fern::size(source, 0) - 2 * fern::radius(kernel);
        size_t const size2 = fern::size(source, 1) - 2 * fern::radius(kernel);
        std::vector<IndexRanges<2>> index_ranges = fern::index_ranges(
            pool.size(), size1, size2);

        for(auto& ranges: index_ranges) {
            // Offset indices by the radius of the kernel.
            ranges = IndexRanges<2>{
                IndexRange(
                    ranges[0].begin() + fern::radius(kernel),
                    ranges[0].end() + fern::radius(kernel)),
                IndexRange(
                    ranges[1].begin() + fern::radius(kernel),
                    ranges[1].end() + fern::radius(kernel))};
            CREATE_INNER_PART_TASK(ranges);
        }

        for(auto& future: futures) {
            future.get();
        }
    }

};

#undef CREATE_BORDER_TASK
#undef CREATE_INNER_PART_TASK

} // namespace dispatch


template<
    class AlternativeForNoDataPolicy,
    class NormalizePolicy,
    class OutOfImagePolicy,
    template<class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class SourceImage,
    class Kernel,
    class DestinationImage
>
void convolve(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    SourceImage const& source,
    Kernel const& kernel,
    DestinationImage& destination)
{
    // Dispatch on execution policy.
    dispatch::Convolve<AlternativeForNoDataPolicy, NormalizePolicy,
        OutOfImagePolicy, OutOfRangePolicy, InputNoDataPolicy,
        OutputNoDataPolicy, SourceImage, Kernel, DestinationImage,
        ExecutionPolicy>::apply(
            input_no_data_policy, output_no_data_policy, execution_policy,
            source, kernel, destination);



    // OutOfImagePolicy:
    // - All convolutions handle out of image values in some way. They may
    //   be skipped, or they may be 'invented', eg: focal average.
    //   There is probably no reason to split algorithms because of this.

    // UseInCaseOfNoData:
    // - All convolutions handle no-data. They may be skipped, or they may be
    //   'invented', eg: focal average. If these values are not skipped, the
    //   process of inventing new values may be expensive. In that case, a
    //   buffer should be used to put all values in before operating the
    //   kernel. This should be optional. The user may know that the input
    //   doesn't contain no-data (check no-data policy!) or only contains
    //   very few no-data values, in which case buffering may cost more than
    //   it saves.

    // Kernels
    // - Cells may contain weights.
    //   - Values are multiplied by the weights.
    //   - Values are normalized using the NormalizePolicy.
    // - Cells may be turned on or off. 'boolean weights'
    //   - Values are just summed. This saves many multiplications.
    //   - Values are normalized using the NormalizePolicy.
    //
    // ----------------------------------------------------------------------
    // KernelTraits<Kernel>::weigh_values triggers a split in the algorithms.
    // ----------------------------------------------------------------------

    // NormalizePolicy:
    // - What to do with the aggregated values that fall within the kernel?
    //   - Divide by the sum of weights.
    //   - Don't divide by the sum of weights.

    // ExecutionPolicy:
    // - SequentialExecution
    // - ParallelExecution
    //
    // ----------------------------------------
    // This triggers a split in the algorithms.
    // ----------------------------------------

    // In case the user wants to use a buffer with preprocessed,
    // there must be a way to fill the buffer with values.
    // This is probably the responsibility of the BufferPolicy. Create
    // overloads with and without this policy. The buffer policy must fill
    // the buffer before we pass it to convolve instead of the source image.

    // // The buffer policy uses:
    // class NormalizePolicy,
    // class OutOfImagePolicy,
    // template<class> class OutOfRangePolicy,
    // class InputNoDataPolicy,

    // // The overload with buffer policy, doesn't use:
    // class ExecutionPolicy, // Handled at a higher level.
    // class OutputNoDataPolicy, // The buffer represents input values.
}

} // namespace detail
} // namespace convolve
} // namespace fern
