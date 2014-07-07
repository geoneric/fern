#pragma once
#include <cassert>
#include <cmath>
#include <cstddef>
#include <utility>
#include <type_traits>
#include "fern/algorithm/convolution/kernel_traits.h"
#include "fern/core/assert.h"
#include "fern/core/argument_traits.h"
#include "fern/core/base_class.h"


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
        InputNoDataPolicy&& input_no_data_policy,
        OutputNoDataPolicy&& output_no_data_policy,
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
        InputNoDataPolicy&& input_no_data_policy,
        OutputNoDataPolicy&& output_no_data_policy,
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
        InputNoDataPolicy&& input_no_data_policy,
        OutputNoDataPolicy&& output_no_data_policy,
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
        InputNoDataPolicy&& input_no_data_policy,
        OutputNoDataPolicy&& output_no_data_policy,
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
        InputNoDataPolicy&& input_no_data_policy,
        OutputNoDataPolicy&& output_no_data_policy,
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
        InputNoDataPolicy&& input_no_data_policy,
        OutputNoDataPolicy&& output_no_data_policy,
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
        InputNoDataPolicy&& input_no_data_policy,
        OutputNoDataPolicy&& output_no_data_policy,
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
        InputNoDataPolicy&& input_no_data_policy,
        OutputNoDataPolicy&& output_no_data_policy,
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
        class OutOfImagePolicy,
        template<class, class> class OutOfRangePolicy,
        class InputNoDataPolicy,
        class OutputNoDataPolicy,
        class SourceImage,
        class Kernel,
        class DestinationImage>
    static void apply(
        InputNoDataPolicy&& input_no_data_policy,
        OutputNoDataPolicy&& output_no_data_policy,
        SourceImage const& source,
        Kernel const& kernel,
        DestinationImage& destination)
    {
        size_t const radius_{radius(kernel)};
        size_t const nr_rows_source{size(source, 0)};
        size_t const nr_cols_source{size(source, 1)};
        size_t first_row_source{0};
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
        for(size_t row_source = radius_; row_source <
                nr_rows_source - radius_; ++row_source) {

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
    InputNoDataPolicy&& input_no_data_policy,
    OutputNoDataPolicy&& output_no_data_policy,
    ExecutionPolicy&& /* execution_policy */,
    SourceImage const& source,
    Kernel const& kernel,
    DestinationImage& destination)
{

    // TODO Specialize based on ExecutionPolicy.


    // Corners.
    dispatch::ConvolveNorthWestCorner<KernelTraits<Kernel>::weigh_values>::
        template apply<AlternativeForNoDataPolicy, NormalizePolicy,
            OutOfImagePolicy, OutOfRangePolicy>(
                std::forward<InputNoDataPolicy>(input_no_data_policy),
                std::forward<OutputNoDataPolicy>(output_no_data_policy),
                source, kernel, destination);
    dispatch::ConvolveNorthEastCorner<KernelTraits<Kernel>::weigh_values>::
        template apply<AlternativeForNoDataPolicy, NormalizePolicy,
            OutOfImagePolicy, OutOfRangePolicy>(
                std::forward<InputNoDataPolicy>(input_no_data_policy),
                std::forward<OutputNoDataPolicy>(output_no_data_policy),
                source, kernel, destination);
    dispatch::ConvolveSouthWestCorner<KernelTraits<Kernel>::weigh_values>::
        template apply<AlternativeForNoDataPolicy, NormalizePolicy,
            OutOfImagePolicy, OutOfRangePolicy>(
                std::forward<InputNoDataPolicy>(input_no_data_policy),
                std::forward<OutputNoDataPolicy>(output_no_data_policy),
                source, kernel, destination);
    dispatch::ConvolveSouthEastCorner<KernelTraits<Kernel>::weigh_values>::
        template apply<AlternativeForNoDataPolicy, NormalizePolicy,
            OutOfImagePolicy, OutOfRangePolicy>(
                std::forward<InputNoDataPolicy>(input_no_data_policy),
                std::forward<OutputNoDataPolicy>(output_no_data_policy),
                source, kernel, destination);

    // Sides.
    dispatch::ConvolveNorthSide<KernelTraits<Kernel>::weigh_values>::
        template apply<AlternativeForNoDataPolicy, NormalizePolicy,
            OutOfImagePolicy, OutOfRangePolicy>(
                std::forward<InputNoDataPolicy>(input_no_data_policy),
                std::forward<OutputNoDataPolicy>(output_no_data_policy),
                source, kernel, destination);
    dispatch::ConvolveWestSide<KernelTraits<Kernel>::weigh_values>::
        template apply<AlternativeForNoDataPolicy, NormalizePolicy,
            OutOfImagePolicy, OutOfRangePolicy>(
                std::forward<InputNoDataPolicy>(input_no_data_policy),
                std::forward<OutputNoDataPolicy>(output_no_data_policy),
                source, kernel, destination);
    dispatch::ConvolveEastSide<KernelTraits<Kernel>::weigh_values>::
        template apply<AlternativeForNoDataPolicy, NormalizePolicy,
            OutOfImagePolicy, OutOfRangePolicy>(
                std::forward<InputNoDataPolicy>(input_no_data_policy),
                std::forward<OutputNoDataPolicy>(output_no_data_policy),
                source, kernel, destination);
    dispatch::ConvolveSouthSide<KernelTraits<Kernel>::weigh_values>::
        template apply<AlternativeForNoDataPolicy, NormalizePolicy,
            OutOfImagePolicy, OutOfRangePolicy>(
                std::forward<InputNoDataPolicy>(input_no_data_policy),
                std::forward<OutputNoDataPolicy>(output_no_data_policy),
                source, kernel, destination);

    // Inner part.
    dispatch::ConvolveInnerPart<KernelTraits<Kernel>::weigh_values>::
        template apply<AlternativeForNoDataPolicy, NormalizePolicy,
            OutOfImagePolicy, OutOfRangePolicy>(
                std::forward<InputNoDataPolicy>(input_no_data_policy),
                std::forward<OutputNoDataPolicy>(output_no_data_policy),
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


/// template<
///     // class ExecutionPolicy,
///     class SourceImage,
///     class Kernel,
///     class DestinationImage,
///     class NormalizePolicy=convolve::DivideByWeights,
///     template<class> class OutOfRangePolicy=nullary::DiscardRangeErrors,
///     class InputNoDataPolicy=SkipNoData,
///     class OutputNoDataPolicy=DontMarkNoData
/// >
/// class Convolve
/// {
/// 
/// public:
/// 
///     using category = focal_operation_tag;
///     using A1 = SourceImage;
///     using A1Value = value_type<A1>;
///     using A2 = Kernel;
///     using A2Value = value_type<A2>;
///     using R = DestinationImage;
///     using RValue = value_type<R>;
/// 
///     FERN_STATIC_ASSERT(std::is_arithmetic, A1Value)
///     FERN_STATIC_ASSERT(std::is_arithmetic, A2Value)
///     FERN_STATIC_ASSERT(std::is_arithmetic, RValue)
/// 
///     FERN_STATIC_ASSERT(std::is_floating_point, A1Value)
///     FERN_STATIC_ASSERT(std::is_floating_point, RValue)
/// 
///     Convolve()
///         : _algorithm()
///     {
///     }
/// 
///     Convolve(
///         InputNoDataPolicy&& input_no_data_policy,  // Universal reference.
///         OutputNoDataPolicy&& output_no_data_policy)  // Universal reference.
///         : _algorithm(
///             std::forward<InputNoDataPolicy>(input_no_data_policy),
///             std::forward<OutputNoDataPolicy>(output_no_data_policy))
///     {
///     }
/// 
///     inline void operator()(
///         A1 const& source_image,
///         A2 const& kernel,
///         R& destination_image)
///     {
///         _algorithm.calculate(source_image, kernel, destination_image);
///     }
/// 
///     // template<
///     //     class Indices>
///     // inline void operator()(
///     //     Indices const& indices,
///     //     A const& values,
///     //     R& result)
///     // {
///     //     _algorithm.calculate(indices, values, result);
///     // }
/// 
/// private:
/// 
///     convolve::detail::dispatch::Convolve<A1, A2, R,
///         NormalizePolicy, OutOfRangePolicy,
///         InputNoDataPolicy, OutputNoDataPolicy,
///         typename base_class<
///             typename ArgumentTraits<A1>::argument_category,
///             array_2d_tag>::type,
///         typename base_class<
///             typename ArgumentTraits<A2>::argument_category,
///             array_2d_tag>::type,
///         typename base_class<
///             typename ArgumentTraits<R>::argument_category,
///             array_2d_tag>::type> _algorithm;
/// 
/// };


///         assert(size(source) == size(destination));
///         assert(size(kernel) > 0);
///         assert(width(kernel) % 2u == 1u);
///         assert(height(kernel) % 2u == 1u);
///         assert(radius(kernel) > 0u);
/// 
///         // Apply kernel on the various sections of the source image.
///         // We divide the source image in 9 section, which differ with
///         // respect to the number of source cells that fall within the
///         // kernel neighborhood:
///         // - 4 Corners: missing cells on two sides.
///         //     - north-west corner
///         //     - north-east corner
///         //     - south-west corner
///         //     - south-east corner
///         // - 4 Sides, between the corners: missing cells on one side.
///         //     - north side
///         //     - west side
///         //     - east side
///         //     - south side
///         // - Center: no missing cells.
///         // For each section of the source image, we call a seperate
///         // function that will handle the section. This code knows exactly
///         // were the missing cells are located. This save us lots of checks
///         // that aren't necessary. For example, when processing the west
///         // side of the source image, the logic can assume that the north,
///         // east and south side of the kernel are located within the source
///         // image. Only at the west side of the kernel some cells are missing.
///     //! Convolve \a source with \a kernel and write the result to \a destination.
///     /*!
///       \tparam    NormalizePolicy Policy used to handle the normalization of the
///                  convolution result by the kernel weights.
///       \tparam    SourceImage Type of source image.
///       \tparam    Kernel Type of convolution kernel.
///       \tparam    DestinationImage Type of destination image.
///       \param     source Source image.
///       \param     kernel Convolution kernel.
///       \param     destination Destination image.
///       \param     row_source Index of row to calculate value for.
///       \param     col_source Index of column to calculate value for.
///       \param     first_row_source Index of first row in source image to use.
///       \param     first_col_source Index of first column in source image to use.
///       \param     first_row_kernel Index of first row in kernel to use.
///       \param     first_col_kernel Index of first column in kernel to use.
///       \param     nr_rows_kernel Number of rows in kernel to use.
///       \param     nr_cols_kernel Number of columns in kernel to use.
/// 
///       This function calculates a value for a single cell in \a destination. This
///       cell is positioned at \a row_source, \a col_source. The new value is
///       calculated based on the values in \a source and the \a kernel weights.
/// 
///       When the kernel is positioned on the cell to calculate a value for, some
///       cells in the \a source may fall outside of the kernel. This means that we
///       may not have values for all cells in the kernel. The caller can configure
///       this by passing \a first_row_source, \a first_col_source,
///       \a first_row_kernel, \a first_col_kernel, \a nr_rows_kernel,
///       \a nr_cols_kernel appropriately. These arguments configure a section of
///       the source and kernel for which the \a source has values to use
///       during the calculation.
///     */
/// 
///         // Verify dimensions of source and destination images are equal
///         assert(size(source, 0) == size(destination, 0));
///         assert(size(source, 1) == size(destination, 1));
/// 
///         assert(nr_rows_kernel > 0);
///         assert(nr_cols_kernel > 0);
/// 
///         // Verify kernel section is positioned within source image.
///         assert(first_row_source < size(source, 0));
///         assert(first_col_source < size(source, 1));
///         assert(first_row_source + nr_rows_kernel <= size(source, 0));
///         assert(first_col_source + nr_cols_kernel <= size(source, 1));
/// 
///         // Verify kernel section is positioned within kernel.
///         assert(first_row_kernel < height(kernel));
///         assert(first_col_kernel < width(kernel));
///         assert(first_row_kernel + nr_rows_kernel <= height(kernel));
///         assert(first_col_kernel + nr_cols_kernel <= width(kernel));
/// 
///         // Verify cell to calculate for is positioned within kernel section of
///         // source image.
///         assert(row_source >= first_row_source);
///         assert(row_source < first_row_source + nr_rows_kernel);
///         assert(col_source >= first_col_source);
///         assert(col_source < first_col_source + nr_cols_kernel);


