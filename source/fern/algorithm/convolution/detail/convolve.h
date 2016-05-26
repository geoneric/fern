// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <type_traits>
#include "fern/algorithm/core/index_ranges.h"
#include "fern/algorithm/core/accumulation_traits.h"
#include "fern/algorithm/convolution/neighborhood/square_traits.h"
#include "fern/algorithm/convolution/kernel_traits.h"
#include "fern/algorithm/policy/execution_policy.h"
#include "fern/core/assert.h"
#include "fern/core/data_type_traits.h"
#include "fern/core/base_class.h"


namespace fern {
namespace algorithm {
namespace convolve {
namespace detail {

template<
    typename Value,
    typename Result>
struct OutOfRangePolicy
{

    FERN_STATIC_ASSERT(std::is_floating_point, Value)
    FERN_STATIC_ASSERT(std::is_floating_point, Result)

    inline static bool within_range(
        Result const& result)
    {
        return std::isfinite(result);
    }

};


namespace dispatch {

template<
    typename SV,
    typename SW,
    typename V,
    typename W>
struct HandleValue
{

    inline static void apply(
        V const& value,
        W const& weight,
        SV& sum_of_values,
        SW& sum_of_weights)
    {
        // Kernel weights are not booleans.
        sum_of_values += value * weight;
        sum_of_weights += weight;
    }

};


template<
    typename SV,
    typename SW,
    typename V>
struct HandleValue<
    SV,
    SW,
    V,
    bool>

{

    inline static void apply(
        V const& value,
        bool const& weight,
        SV& sum_of_values,
        SW& sum_of_weights)
    {
        // Kernel weights are booleans.
        if(weight) {
            sum_of_values += value;
            sum_of_weights += 1;
        }
    }

};


struct ConvolveNorthWestCorner
{

    template<
        typename AlternativeForNoDataPolicy,
        typename NormalizePolicy,
        typename OutOfImagePolicy,
        template<typename, typename> class OutOfRangePolicy,
        typename InputNoDataPolicy,
        typename OutputNoDataPolicy,
        typename SourceImage,
        typename Kernel,
        typename DestinationImage>
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

        using V = value_type<SourceImage>;
        using SV = accumulate_type<V>;
        using W = value_type<Kernel>;
        using SW = accumulate_type<W>;

        SV sum_of_values;
        SW sum_of_weights;
        V out_of_image_value;
        V alternative_value;

        using OORP = OutOfRangePolicy<value_type<SourceImage>,
            value_type<DestinationImage>>;
        using NP = NormalizePolicy;
        using AFNP = AlternativeForNoDataPolicy;

        bool value_seen;

        size_t index_;

        // Loop over all cells that are situated in the north west corner
        // and for which some cells are outside of the kernel.
        for(size_t row_source = 0; row_source < radius_; ++row_source) {

            first_col_kernel = radius_;
            nr_cols_kernel = radius_ + 1;
            nr_cols_outside_of_image = radius_;

            index_ = index(source, row_source, 0);

            for(size_t col_source = 0; col_source < radius_; ++col_source) {

                sum_of_values = AccumulationTraits<V>::zero;
                sum_of_weights = AccumulationTraits<W>::zero;
                value_seen = false;

                if(std::get<0>(input_no_data_policy).is_no_data(index_)) {
                    output_no_data_policy.mark_as_no_data(index_);
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
                    {
                        size_t kernel_index;
                        for(size_t out_of_image_kernel_row = 0;
                                out_of_image_kernel_row < size(kernel, 0);
                                ++out_of_image_kernel_row) {
                            kernel_index = index(kernel,
                                out_of_image_kernel_row, 0);
                            for(size_t out_of_image_kernel_col = 0;
                                    out_of_image_kernel_col < size(kernel, 1);
                                        ++out_of_image_kernel_col) {
                                if(out_of_image_kernel_row < first_row_kernel ||
                                    out_of_image_kernel_col < first_col_kernel)
                                {
                                    if(OutOfImagePolicy::value_north_west(
                                            input_no_data_policy,
                                            source,
                                            out_of_image_kernel_row,
                                            out_of_image_kernel_col,
                                            first_row_kernel, first_col_kernel,
                                            nr_rows_kernel, nr_cols_kernel,
                                            first_row_source, first_col_source,
                                            out_of_image_value)) {
                                        HandleValue<SV, SW, V, W>::apply(
                                            out_of_image_value,
                                            get(kernel, kernel_index),
                                            sum_of_values, sum_of_weights);
                                        value_seen = true;
                                    }
                                }

                                ++kernel_index;
                            }
                        }
                    }

                    // Handle cells positioned in the kernel, and in the source
                    // image.
                    size_t source_index, kernel_index;
                    for(size_t row = 0; row < nr_rows_kernel; ++row) {
                        source_index = index(source, first_row_source + row,
                            first_col_source + 0);
                        kernel_index = index(kernel, first_row_kernel + row,
                            first_col_kernel + 0);
                        for(size_t col = 0; col < nr_cols_kernel; ++col) {
                            if(std::get<0>(input_no_data_policy).is_no_data(
                                    source_index)) {
                                if(AFNP::value(
                                        input_no_data_policy,
                                        source,
                                        size(source, 0),
                                        size(source, 1),
                                        first_row_source + row,
                                        first_col_source + col,
                                        alternative_value)) {
                                    HandleValue<SV, SW, V, W>::apply(
                                        alternative_value,
                                        get(kernel, kernel_index),
                                        sum_of_values, sum_of_weights);
                                    value_seen = true;
                                }
                            }
                            else {
                                HandleValue<SV, SW, V, W>::apply(
                                    get(source, source_index),
                                    get(kernel, kernel_index),
                                    sum_of_values, sum_of_weights);
                                value_seen = true;
                            }

                            ++source_index;
                            ++kernel_index;
                        }
                    }
                }


                // The result can go out of range when the convolution
                // results in an infinite value. Normalizing such a
                // value makes no sense.

                // TODO OutOfRangePolicy must handle integral results too.
                if(!value_seen || !OORP::within_range(sum_of_values)) {
                    output_no_data_policy.mark_as_no_data(index_);
                }
                else {
                    get(destination, index_) = NP::normalize(sum_of_values,
                        sum_of_weights);
                }

                --first_col_kernel;
                ++nr_cols_kernel;
                --nr_cols_outside_of_image;
                ++index_;
            }

            --first_row_kernel;
            ++nr_rows_kernel;
            --nr_rows_outside_of_image;
        }
    }

};


struct ConvolveNorthEastCorner
{

    template<
        typename AlternativeForNoDataPolicy,
        typename NormalizePolicy,
        typename OutOfImagePolicy,
        template<typename, typename> class OutOfRangePolicy,
        typename InputNoDataPolicy,
        typename OutputNoDataPolicy,
        typename SourceImage,
        typename Kernel,
        typename DestinationImage>
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

        using V = value_type<SourceImage>;
        using SV = accumulate_type<V>;
        using W = value_type<Kernel>;
        using SW = accumulate_type<W>;

        SV sum_of_values = AccumulationTraits<V>::zero;
        SW sum_of_weights = AccumulationTraits<W>::zero;
        V out_of_image_value;
        V alternative_value;

        using OORP = OutOfRangePolicy<value_type<SourceImage>,
            value_type<DestinationImage>>;
        using NP = NormalizePolicy;
        using AFNP = AlternativeForNoDataPolicy;

        bool value_seen;

        size_t index_;

        // Loop over all cells that are situated in the north east corner
        // and for which some cells are outside of the kernel.
        for(size_t row_source = 0; row_source < radius_; ++row_source) {

            first_col_source = nr_cols_source - radius_ - radius_;
            nr_cols_kernel = radius_ + radius_;
            nr_cols_outside_of_image = 1;

            index_ = index(source, row_source, nr_cols_source - radius_);

            for(size_t col_source = nr_cols_source - radius_;
                    col_source < nr_cols_source; ++col_source) {

                sum_of_values = AccumulationTraits<V>::zero;
                sum_of_weights = AccumulationTraits<W>::zero;
                value_seen = false;

                if(std::get<0>(input_no_data_policy).is_no_data(index_)) {
                    output_no_data_policy.mark_as_no_data(index_);
                }
                else {
                    // Handle cells positioned in the kernel, but outside of
                    // the source image.
                    {
                        size_t kernel_index;
                        for(size_t out_of_image_kernel_row = 0;
                                out_of_image_kernel_row < size(kernel, 0);
                                ++out_of_image_kernel_row) {
                            kernel_index = index(kernel,
                                out_of_image_kernel_row, 0);
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
                                        HandleValue<SV, SW, V, W>::apply(
                                            out_of_image_value,
                                            get(kernel, kernel_index),
                                            sum_of_values, sum_of_weights);
                                        value_seen = true;
                                    }
                                }

                                ++kernel_index;
                            }
                        }
                    }

                    // Handle cells positioned in the kernel, and in the source
                    // image.
                    size_t source_index, kernel_index;
                    for(size_t row = 0; row < nr_rows_kernel; ++row) {
                        source_index = index(source, first_row_source + row,
                            first_col_source + 0);
                        kernel_index = index(kernel, first_row_kernel + row,
                            first_col_kernel + 0);
                        for(size_t col = 0; col < nr_cols_kernel; ++col) {
                            if(std::get<0>(input_no_data_policy).is_no_data(
                                    source_index)) {
                                if(AFNP::value(
                                        input_no_data_policy,
                                        source,
                                        size(source, 0),
                                        size(source, 1),
                                        first_row_source + row,
                                        first_col_source + col,
                                        alternative_value)) {
                                    HandleValue<SV, SW, V, W>::apply(
                                        alternative_value,
                                        get(kernel, kernel_index),
                                        sum_of_values, sum_of_weights);
                                    value_seen = true;
                                }
                            }
                            else {
                                HandleValue<SV, SW, V, W>::apply(
                                    get(source, source_index),
                                    get(kernel, kernel_index),
                                    sum_of_values, sum_of_weights);
                                value_seen = true;
                            }

                            ++source_index;
                            ++kernel_index;
                        }
                    }
                }


                // The result can go out of range when the convolution
                // results in an infinite value. Normalizing such a
                // value makes no sense.

                // TODO OutOfRangePolicy must handle integral results too.
                if(!value_seen || !OORP::within_range(sum_of_values)) {
                    output_no_data_policy.mark_as_no_data(index_);
                }
                else {
                    get(destination, index_) = NP::normalize(sum_of_values,
                        sum_of_weights);
                }

                ++first_col_source;
                --nr_cols_kernel;
                ++nr_cols_outside_of_image;
                ++index_;
            }

            --first_row_kernel;
            ++nr_rows_kernel;
            --nr_rows_outside_of_image;
        }
    }

};


struct ConvolveSouthWestCorner
{

    template<
        typename AlternativeForNoDataPolicy,
        typename NormalizePolicy,
        typename OutOfImagePolicy,
        template<typename, typename> class OutOfRangePolicy,
        typename InputNoDataPolicy,
        typename OutputNoDataPolicy,
        typename SourceImage,
        typename Kernel,
        typename DestinationImage>
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
        size_t nr_rows_outside_of_image{0};
        size_t nr_cols_outside_of_image;

        using V = value_type<SourceImage>;
        using SV = accumulate_type<V>;
        using W = value_type<Kernel>;
        using SW = accumulate_type<W>;

        SV sum_of_values;
        SW sum_of_weights;
        V out_of_image_value;
        V alternative_value;

        using OORP = OutOfRangePolicy<value_type<SourceImage>,
            value_type<DestinationImage>>;
        using NP = NormalizePolicy;
        using AFNP = AlternativeForNoDataPolicy;

        bool value_seen;

        size_t index_;

        // Loop over all cells that are situated in the south west corner
        // and for which some cells are outside of the kernel.
        for(size_t row_source = nr_rows_source - radius_;
                row_source < nr_rows_source; ++row_source) {

            first_col_kernel = radius_;
            nr_cols_kernel = radius_ + 1;
            nr_cols_outside_of_image = radius_;

            index_ = index(source, row_source, 0);

            for(size_t col_source = 0; col_source < radius_; ++col_source) {

                sum_of_values = AccumulationTraits<V>::zero;
                sum_of_weights = AccumulationTraits<W>::zero;
                value_seen = false;

                if(std::get<0>(input_no_data_policy).is_no_data(index_)) {
                    output_no_data_policy.mark_as_no_data(index_);
                }
                else {
                    // Handle cells positioned in the kernel, but outside of
                    // the source image.
                    {
                        size_t kernel_index;
                        for(size_t out_of_image_kernel_row = 0;
                                out_of_image_kernel_row < size(kernel, 0);
                                ++out_of_image_kernel_row) {
                            kernel_index = index(kernel,
                                out_of_image_kernel_row, 0);
                            for(size_t out_of_image_kernel_col = 0;
                                    out_of_image_kernel_col < size(kernel, 1);
                                        ++out_of_image_kernel_col) {
                                if(out_of_image_kernel_row >= nr_rows_kernel ||
                                    out_of_image_kernel_col < first_col_kernel)
                                {

                                    if(OutOfImagePolicy::value_south_west(
                                            input_no_data_policy,
                                            source,
                                            out_of_image_kernel_row,
                                            out_of_image_kernel_col,
                                            first_row_kernel, first_col_kernel,
                                            nr_rows_kernel, nr_cols_kernel,
                                            first_row_source, first_col_source,
                                            out_of_image_value)) {
                                        HandleValue<SV, SW, V, W>::apply(
                                            out_of_image_value,
                                            get(kernel, kernel_index),
                                            sum_of_values, sum_of_weights);
                                        value_seen = true;
                                    }
                                }

                                ++kernel_index;
                            }
                        }
                    }

                    // Handle cells positioned in the kernel, and in the source
                    // image.
                    size_t source_index, kernel_index;
                    for(size_t row = 0; row < nr_rows_kernel; ++row) {
                        source_index = index(source, first_row_source + row,
                            first_col_source + 0);
                        kernel_index = index(kernel, first_row_kernel + row,
                            first_col_kernel + 0);
                        for(size_t col = 0; col < nr_cols_kernel; ++col) {
                            if(std::get<0>(input_no_data_policy).is_no_data(
                                    source_index)) {
                                if(AFNP::value(
                                        input_no_data_policy,
                                        source,
                                        size(source, 0),
                                        size(source, 1),
                                        first_row_source + row,
                                        first_col_source + col,
                                        alternative_value)) {
                                    HandleValue<SV, SW, V, W>::apply(
                                        alternative_value,
                                        get(kernel, kernel_index),
                                        sum_of_values, sum_of_weights);
                                    value_seen = true;
                                }
                            }
                            else {
                                HandleValue<SV, SW, V, W>::apply(
                                    get(source, source_index),
                                    get(kernel, kernel_index),
                                    sum_of_values, sum_of_weights);
                                value_seen = true;
                            }

                            ++source_index;
                            ++kernel_index;
                        }
                    }
                }


                // The result can go out of range when the convolution
                // results in an infinite value. Normalizing such a
                // value makes no sense.

                // TODO OutOfRangePolicy must handle integral results too.
                if(!value_seen || !OORP::within_range(sum_of_values)) {
                    output_no_data_policy.mark_as_no_data(index_);
                }
                else {
                    get(destination, index_) = NP::normalize(sum_of_values,
                        sum_of_weights);
                }

                --first_col_kernel;
                ++nr_cols_kernel;
                --nr_cols_outside_of_image;
                ++index_;
            }

            ++first_row_source;
            --nr_rows_kernel;
            ++nr_rows_outside_of_image;
        }
    }

};


struct ConvolveSouthEastCorner
{

    template<
        typename AlternativeForNoDataPolicy,
        typename NormalizePolicy,
        typename OutOfImagePolicy,
        template<typename, typename> class OutOfRangePolicy,
        typename InputNoDataPolicy,
        typename OutputNoDataPolicy,
        typename SourceImage,
        typename Kernel,
        typename DestinationImage>
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
        size_t nr_rows_outside_of_image{0};
        size_t nr_cols_outside_of_image;

        using V = value_type<SourceImage>;
        using SV = accumulate_type<V>;
        using W = value_type<Kernel>;
        using SW = accumulate_type<W>;

        SV sum_of_values;
        SW sum_of_weights;
        V out_of_image_value;
        V alternative_value;

        using OORP = OutOfRangePolicy<value_type<SourceImage>,
            value_type<DestinationImage>>;
        using NP = NormalizePolicy;
        using AFNP = AlternativeForNoDataPolicy;

        bool value_seen;

        size_t index_;

        // Loop over all cells that are situated in the south west corner
        // and for which some cells are outside of the kernel.
        for(size_t row_source = nr_rows_source - radius_;
                row_source < nr_rows_source; ++row_source) {

            first_col_source = nr_cols_source - radius_ - radius_;
            nr_cols_kernel = radius_ + radius_;
            nr_cols_outside_of_image = 1;

            index_ = index(source, row_source, nr_cols_source - radius_);

            for(size_t col_source = nr_cols_source - radius_;
                    col_source < nr_cols_source; ++col_source) {

                sum_of_values = AccumulationTraits<V>::zero;
                sum_of_weights = AccumulationTraits<W>::zero;
                value_seen = false;

                if(std::get<0>(input_no_data_policy).is_no_data(index_)) {
                    output_no_data_policy.mark_as_no_data(index_);
                }
                else {
                    // Handle cells positioned in the kernel, but outside of
                    // the source image.
                    {
                        size_t kernel_index;
                        for(size_t out_of_image_kernel_row = 0;
                                out_of_image_kernel_row < size(kernel, 0);
                                ++out_of_image_kernel_row) {
                            kernel_index = index(kernel,
                                out_of_image_kernel_row, 0);
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
                                        HandleValue<SV, SW, V, W>::apply(
                                            out_of_image_value,
                                            get(kernel, kernel_index),
                                            sum_of_values, sum_of_weights);
                                        value_seen = true;
                                    }
                                }

                                ++kernel_index;
                            }
                        }
                    }

                    // Handle cells positioned in the kernel, and in the source
                    // image.
                    size_t source_index, kernel_index;
                    for(size_t row = 0; row < nr_rows_kernel; ++row) {
                        source_index = index(source, first_row_source + row,
                            first_col_source + 0);
                        kernel_index = index(kernel, first_row_kernel + row,
                            first_col_kernel + 0);
                        for(size_t col = 0; col < nr_cols_kernel; ++col) {
                            if(std::get<0>(input_no_data_policy).is_no_data(
                                    source_index)) {
                                if(AFNP::value(
                                        input_no_data_policy,
                                        source,
                                        size(source, 0),
                                        size(source, 1),
                                        first_row_source + row,
                                        first_col_source + col,
                                        alternative_value)) {
                                    HandleValue<SV, SW, V, W>::apply(
                                        alternative_value,
                                        get(kernel, kernel_index),
                                        sum_of_values, sum_of_weights);
                                    value_seen = true;
                                }
                            }
                            else {
                                HandleValue<SV, SW, V, W>::apply(
                                    get(source, source_index),
                                    get(kernel, kernel_index),
                                    sum_of_values, sum_of_weights);
                                value_seen = true;
                            }

                            ++source_index;
                            ++kernel_index;
                        }
                    }
                }


                // The result can go out of range when the convolution
                // results in an infinite value. Normalizing such a
                // value makes no sense.

                // TODO OutOfRangePolicy must handle integral results too.
                if(!value_seen || !OORP::within_range(sum_of_values)) {
                    output_no_data_policy.mark_as_no_data(index_);
                }
                else {
                    get(destination, index_) = NP::normalize(sum_of_values,
                        sum_of_weights);
                }

                ++first_col_source;
                --nr_cols_kernel;
                ++nr_cols_outside_of_image;
                ++index_;
            }

            ++first_row_source;
            --nr_rows_kernel;
            ++nr_rows_outside_of_image;
        }
    }

};


struct ConvolveNorthSide
{

    template<
        typename AlternativeForNoDataPolicy,
        typename NormalizePolicy,
        typename OutOfImagePolicy,
        template<typename, typename> class OutOfRangePolicy,
        typename InputNoDataPolicy,
        typename OutputNoDataPolicy,
        typename SourceImage,
        typename Kernel,
        typename DestinationImage>
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

        using V = value_type<SourceImage>;
        using SV = accumulate_type<V>;
        using W = value_type<Kernel>;
        using SW = accumulate_type<W>;

        SV sum_of_values;
        SW sum_of_weights;
        V out_of_image_value;
        V alternative_value;

        using OORP = OutOfRangePolicy<value_type<SourceImage>,
            value_type<DestinationImage>>;
        using NP = NormalizePolicy;
        using AFNP = AlternativeForNoDataPolicy;

        bool value_seen;

        size_t index_;

        // Loop over all cells that are situated in the north side and for
        // which some cells are outside of the kernel.
        for(size_t row_source = 0; row_source < radius_; ++row_source) {

            first_col_source = 0;

            index_ = index(source, row_source, radius_);

            for(size_t col_source = radius_; col_source <
                    nr_cols_source - radius_; ++col_source) {

                sum_of_values = AccumulationTraits<V>::zero;
                sum_of_weights = AccumulationTraits<W>::zero;
                value_seen = false;

                if(std::get<0>(input_no_data_policy).is_no_data(index_)) {
                    output_no_data_policy.mark_as_no_data(index_);
                }
                else {
                    // Handle cells positioned in the kernel, but outside of
                    // the source image.
                    {
                        size_t kernel_index;
                        for(size_t out_of_image_kernel_row = 0;
                                out_of_image_kernel_row < size(kernel, 0);
                                ++out_of_image_kernel_row) {
                            kernel_index = index(kernel,
                                out_of_image_kernel_row, 0);
                            for(size_t out_of_image_kernel_col = 0;
                                    out_of_image_kernel_col < size(kernel, 1);
                                    ++out_of_image_kernel_col) {

                                assert(out_of_image_kernel_col >=
                                    first_col_kernel);

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
                                        HandleValue<SV, SW, V, W>::apply(
                                            out_of_image_value,
                                            get(kernel, kernel_index),
                                            sum_of_values, sum_of_weights);
                                        value_seen = true;
                                    }
                                }

                                ++kernel_index;
                            }
                        }
                    }

                    // Handle cells positioned in the kernel, and in the source
                    // image.
                    size_t source_index, kernel_index;
                    for(size_t row = 0; row < nr_rows_kernel; ++row) {
                        source_index = index(source, first_row_source + row,
                            first_col_source + 0);
                        kernel_index = index(kernel, first_row_kernel + row,
                            first_col_kernel + 0);
                        for(size_t col = 0; col < nr_cols_kernel; ++col) {
                            if(std::get<0>(input_no_data_policy).is_no_data(
                                    source_index)) {
                                if(AFNP::value(
                                        input_no_data_policy,
                                        source,
                                        size(source, 0),
                                        size(source, 1),
                                        first_row_source + row,
                                        first_col_source + col,
                                        alternative_value)) {
                                    HandleValue<SV, SW, V, W>::apply(
                                        alternative_value,
                                        get(kernel, kernel_index),
                                        sum_of_values, sum_of_weights);
                                    value_seen = true;
                                }
                            }
                            else {
                                HandleValue<SV, SW, V, W>::apply(
                                    get(source, source_index),
                                    get(kernel, kernel_index),
                                    sum_of_values, sum_of_weights);
                                value_seen = true;
                            }

                            ++source_index;
                            ++kernel_index;
                        }
                    }
                }


                // The result can go out of range when the convolution
                // results in an infinite value. Normalizing such a
                // value makes no sense.

                // TODO OutOfRangePolicy must handle integral results too.
                if(!value_seen || !OORP::within_range(sum_of_values)) {
                    output_no_data_policy.mark_as_no_data(index_);
                }
                else {
                    get(destination, index_) = NP::normalize(sum_of_values,
                        sum_of_weights);
                }

                ++first_col_source;
                ++index_;
            }

            --first_row_kernel;
            ++nr_rows_kernel;
            --nr_rows_outside_of_image;
        }
    }

};


struct ConvolveWestSide
{

    template<
        typename AlternativeForNoDataPolicy,
        typename NormalizePolicy,
        typename OutOfImagePolicy,
        template<typename, typename> class OutOfRangePolicy,
        typename InputNoDataPolicy,
        typename OutputNoDataPolicy,
        typename SourceImage,
        typename Kernel,
        typename DestinationImage>
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

        using V = value_type<SourceImage>;
        using SV = accumulate_type<V>;
        using W = value_type<Kernel>;
        using SW = accumulate_type<W>;

        SV sum_of_values;
        SW sum_of_weights;
        V out_of_image_value;
        V alternative_value;

        using OORP = OutOfRangePolicy<value_type<SourceImage>,
            value_type<DestinationImage>>;
        using NP = NormalizePolicy;
        using AFNP = AlternativeForNoDataPolicy;

        bool value_seen;

        size_t index_;

        // Loop over all cells that are situated in the west side and for
        // which some cells are outside of the kernel.
        for(size_t row_source = radius_; row_source <
                nr_rows_source - radius_; ++row_source) {

            first_col_kernel = radius_;
            nr_cols_kernel = radius_ + 1;
            nr_cols_outside_of_image = radius_;

            index_ = index(source, row_source, 0);

            for(size_t col_source = 0; col_source < radius_; ++col_source) {

                sum_of_values = AccumulationTraits<V>::zero;
                sum_of_weights = AccumulationTraits<W>::zero;
                value_seen = false;

                if(std::get<0>(input_no_data_policy).is_no_data(index_)) {
                    output_no_data_policy.mark_as_no_data(index_);
                }
                else {
                    // Handle cells positioned in the kernel, but outside of
                    // the source image.
                    {
                        size_t kernel_index;
                        for(size_t out_of_image_kernel_row = 0;
                                out_of_image_kernel_row < size(kernel, 0);
                                ++out_of_image_kernel_row) {
                            kernel_index = index(kernel,
                                out_of_image_kernel_row, 0);
                            for(size_t out_of_image_kernel_col = 0;
                                    out_of_image_kernel_col < size(kernel, 1);
                                    ++out_of_image_kernel_col) {

                                assert(out_of_image_kernel_row >=
                                    first_row_kernel);

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
                                        HandleValue<SV, SW, V, W>::apply(
                                            out_of_image_value,
                                            get(kernel, kernel_index),
                                            sum_of_values, sum_of_weights);
                                        value_seen = true;
                                    }
                                }

                                ++kernel_index;
                            }
                        }
                    }

                    // Handle cells positioned in the kernel, and in the source
                    // image.
                    size_t source_index, kernel_index;
                    for(size_t row = 0; row < nr_rows_kernel; ++row) {
                        source_index = index(source, first_row_source + row,
                            first_col_source + 0);
                        kernel_index = index(kernel, first_row_kernel + row,
                            first_col_kernel + 0);
                        for(size_t col = 0; col < nr_cols_kernel; ++col) {
                            if(std::get<0>(input_no_data_policy).is_no_data(
                                    source_index)) {
                                if(AFNP::value(
                                        input_no_data_policy,
                                        source,
                                        size(source, 0),
                                        size(source, 1),
                                        first_row_source + row,
                                        first_col_source + col,
                                        alternative_value)) {
                                    HandleValue<SV, SW, V, W>::apply(
                                        alternative_value,
                                        get(kernel, kernel_index),
                                        sum_of_values, sum_of_weights);
                                    value_seen = true;
                                }
                            }
                            else {
                                HandleValue<SV, SW, V, W>::apply(
                                    get(source, source_index),
                                    get(kernel, kernel_index),
                                    sum_of_values, sum_of_weights);
                                value_seen = true;
                            }

                            ++source_index;
                            ++kernel_index;
                        }
                    }
                }


                // The result can go out of range when the convolution
                // results in an infinite value. Normalizing such a
                // value makes no sense.

                // TODO OutOfRangePolicy must handle integral results too.
                if(!value_seen || !OORP::within_range(sum_of_values)) {
                    output_no_data_policy.mark_as_no_data(index_);
                }
                else {
                    get(destination, index_) = NP::normalize(sum_of_values,
                        sum_of_weights);
                }

                --first_col_kernel;
                ++nr_cols_kernel;
                --nr_cols_outside_of_image;
                ++index_;
            }

            ++first_row_source;
        }
    }

};


struct ConvolveEastSide
{

    template<
        typename AlternativeForNoDataPolicy,
        typename NormalizePolicy,
        typename OutOfImagePolicy,
        template<typename, typename> class OutOfRangePolicy,
        typename InputNoDataPolicy,
        typename OutputNoDataPolicy,
        typename SourceImage,
        typename Kernel,
        typename DestinationImage>
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

        using V = value_type<SourceImage>;
        using SV = accumulate_type<V>;
        using W = value_type<Kernel>;
        using SW = accumulate_type<W>;

        SV sum_of_values;
        SW sum_of_weights;
        V out_of_image_value;
        V alternative_value;

        using OORP = OutOfRangePolicy<value_type<SourceImage>,
            value_type<DestinationImage>>;
        using NP = NormalizePolicy;
        using AFNP = AlternativeForNoDataPolicy;

        bool value_seen;

        size_t index_;

        // Loop over all cells that are situated in the east side and for
        // which some cells are outside of the kernel.
        for(size_t row_source = radius_; row_source <
                nr_rows_source - radius_; ++row_source) {

            first_col_source = nr_cols_source - radius_ - radius_;
            nr_cols_kernel = radius_ + radius_;
            nr_cols_outside_of_image = 1;

            index_ = index(source, row_source, nr_cols_source - radius_);

            for(size_t col_source = nr_cols_source - radius_;
                    col_source < nr_cols_source; ++col_source) {

                sum_of_values = AccumulationTraits<V>::zero;
                sum_of_weights = AccumulationTraits<W>::zero;
                value_seen = false;

                if(std::get<0>(input_no_data_policy).is_no_data(index_)) {
                    output_no_data_policy.mark_as_no_data(index_);
                }
                else {
                    // Handle cells positioned in the kernel, but outside of
                    // the source image.
                    {
                        size_t kernel_index;
                        for(size_t out_of_image_kernel_row = 0;
                                out_of_image_kernel_row < size(kernel, 0);
                                ++out_of_image_kernel_row) {
                            kernel_index = index(kernel,
                                out_of_image_kernel_row, 0);
                            for(size_t out_of_image_kernel_col = 0;
                                    out_of_image_kernel_col < size(kernel, 1);
                                    ++out_of_image_kernel_col) {

                                assert(out_of_image_kernel_row >=
                                    first_row_kernel);

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
                                        HandleValue<SV, SW, V, W>::apply(
                                            out_of_image_value,
                                            get(kernel, kernel_index),
                                            sum_of_values, sum_of_weights);
                                        value_seen = true;
                                    }
                                }

                                ++kernel_index;
                            }
                        }
                    }

                    // Handle cells positioned in the kernel, and in the source
                    // image.
                    size_t source_index, kernel_index;
                    for(size_t row = 0; row < nr_rows_kernel; ++row) {
                        source_index = index(source, first_row_source + row,
                            first_col_source + 0);
                        kernel_index = index(kernel, first_row_kernel + row,
                            first_col_kernel + 0);
                        for(size_t col = 0; col < nr_cols_kernel; ++col) {
                            if(std::get<0>(input_no_data_policy).is_no_data(
                                    source_index)) {
                                if(AFNP::value(
                                        input_no_data_policy,
                                        source,
                                        size(source, 0),
                                        size(source, 1),
                                        first_row_source + row,
                                        first_col_source + col,
                                        alternative_value)) {
                                    HandleValue<SV, SW, V, W>::apply(
                                        alternative_value,
                                        get(kernel, kernel_index),
                                        sum_of_values, sum_of_weights);
                                    value_seen = true;
                                }
                            }
                            else {
                                HandleValue<SV, SW, V, W>::apply(
                                    get(source, source_index),
                                    get(kernel, kernel_index),
                                    sum_of_values, sum_of_weights);
                                value_seen = true;
                            }

                            ++source_index;
                            ++kernel_index;
                        }
                    }
                }


                // The result can go out of range when the convolution
                // results in an infinite value. Normalizing such a
                // value makes no sense.

                // TODO OutOfRangePolicy must handle integral results too.
                if(!value_seen || !OORP::within_range(sum_of_values)) {
                    output_no_data_policy.mark_as_no_data(index_);
                }
                else {
                    get(destination, index_) = NP::normalize(sum_of_values,
                        sum_of_weights);
                }

                ++first_col_source;
                --nr_cols_kernel;
                ++nr_cols_outside_of_image;
                ++index_;
            }

            ++first_row_source;
        }
    }

};


struct ConvolveSouthSide
{

    template<
        typename AlternativeForNoDataPolicy,
        typename NormalizePolicy,
        typename OutOfImagePolicy,
        template<typename, typename> class OutOfRangePolicy,
        typename InputNoDataPolicy,
        typename OutputNoDataPolicy,
        typename SourceImage,
        typename Kernel,
        typename DestinationImage>
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

        using V = value_type<SourceImage>;
        using SV = accumulate_type<V>;
        using W = value_type<Kernel>;
        using SW = accumulate_type<W>;

        SV sum_of_values;
        SW sum_of_weights;
        V out_of_image_value;
        V alternative_value;

        using OORP = OutOfRangePolicy<value_type<SourceImage>,
            value_type<DestinationImage>>;
        using NP = NormalizePolicy;
        using AFNP = AlternativeForNoDataPolicy;

        bool value_seen;

        size_t index_;

        // Loop over all cells that are situated in the south side and for
        // which some cells are outside of the kernel.
        for(size_t row_source = nr_rows_source - radius_;
                row_source < nr_rows_source; ++row_source) {

            first_col_source = 0;

            index_ = index(source, row_source, radius_);

            for(size_t col_source = radius_; col_source <
                    nr_cols_source - radius_; ++col_source) {

                sum_of_values = AccumulationTraits<V>::zero;
                sum_of_weights = AccumulationTraits<W>::zero;
                value_seen = false;

                if(std::get<0>(input_no_data_policy).is_no_data(index_)) {
                    output_no_data_policy.mark_as_no_data(index_);
                }
                else {
                    // Handle cells positioned in the kernel, but outside of
                    // the source image.
                    {
                        size_t kernel_index;
                        for(size_t out_of_image_kernel_row = 0;
                                out_of_image_kernel_row < size(kernel, 0);
                                ++out_of_image_kernel_row) {
                            kernel_index = index(kernel,
                                out_of_image_kernel_row, 0);
                            for(size_t out_of_image_kernel_col = 0;
                                    out_of_image_kernel_col < size(kernel, 1);
                                    ++out_of_image_kernel_col) {

                                assert(out_of_image_kernel_col >=
                                    first_col_kernel);

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
                                        HandleValue<SV, SW, V, W>::apply(
                                            out_of_image_value,
                                            get(kernel, kernel_index),
                                            sum_of_values, sum_of_weights);
                                        value_seen = true;
                                    }
                                }

                                ++kernel_index;
                            }
                        }
                    }

                    // Handle cells positioned in the kernel, and in the source
                    // image.
                    size_t source_index, kernel_index;
                    for(size_t row = 0; row < nr_rows_kernel; ++row) {
                        source_index = index(source, first_row_source + row,
                            first_col_source + 0);
                        kernel_index = index(kernel, first_row_kernel + row,
                            first_col_kernel + 0);
                        for(size_t col = 0; col < nr_cols_kernel; ++col) {
                            if(std::get<0>(input_no_data_policy).is_no_data(
                                    source_index)) {
                                if(AFNP::value(
                                        input_no_data_policy,
                                        source,
                                        size(source, 0),
                                        size(source, 1),
                                        first_row_source + row,
                                        first_col_source + col,
                                        alternative_value)) {
                                    HandleValue<SV, SW, V, W>::apply(
                                        alternative_value,
                                        get(kernel, kernel_index),
                                        sum_of_values, sum_of_weights);
                                    value_seen = true;
                                }
                            }
                            else {
                                HandleValue<SV, SW, V, W>::apply(
                                    get(source, source_index),
                                    get(kernel, kernel_index),
                                    sum_of_values, sum_of_weights);
                                value_seen = true;
                            }

                            ++source_index;
                            ++kernel_index;
                        }
                    }
                }


                // The result can go out of range when the convolution
                // results in an infinite value. Normalizing such a
                // value makes no sense.

                // TODO OutOfRangePolicy must handle integral results too.
                if(!value_seen || !OORP::within_range(sum_of_values)) {
                    output_no_data_policy.mark_as_no_data(index_);
                }
                else {
                    get(destination, index_) = NP::normalize(sum_of_values,
                        sum_of_weights);
                }

                ++first_col_source;
                ++index_;
            }

            // --first_row_kernel;
            ++first_row_source;
            --nr_rows_kernel;
            ++nr_rows_outside_of_image;
        }
    }

};


template<
    bool weigh_values
>
struct ConvolveInnerPart;


template<>
struct ConvolveInnerPart<true>
{

    template<
        typename AlternativeForNoDataPolicy,
        typename NormalizePolicy,
        template<typename, typename> class OutOfRangePolicy,
        typename InputNoDataPolicy,
        typename OutputNoDataPolicy,
        typename SourceImage,
        typename Kernel,
        typename DestinationImage>
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

        using V = value_type<SourceImage>;
        using SV = accumulate_type<V>;
        using W = value_type<Kernel>;
        using SW = accumulate_type<W>;

        SV sum_of_values;
        SW sum_of_weights;
        V alternative_value;

        using OORP = OutOfRangePolicy<value_type<SourceImage>,
            value_type<DestinationImage>>;
        using NP = NormalizePolicy;
        using AFNP = AlternativeForNoDataPolicy;

        bool value_seen;

        size_t index_;

        // Loop over all cells that are situated in the inner part. The kernel
        // does not extent outside of the source image.
        for(size_t row_source = index_ranges[0].begin(); row_source <
                index_ranges[0].end(); ++row_source) {

            first_col_source = index_ranges[1].begin() - radius_;

            index_ = index(source, row_source, index_ranges[1].begin());

            for(size_t col_source = index_ranges[1].begin(); col_source <
                    index_ranges[1].end(); ++col_source) {

                sum_of_values = AccumulationTraits<V>::zero;
                sum_of_weights = AccumulationTraits<W>::zero;
                value_seen = false;

                if(std::get<0>(input_no_data_policy).is_no_data(index_)) {
                    output_no_data_policy.mark_as_no_data(index_);
                }
                else {
                    // Handle cells positioned in the kernel, and in the source
                    // image.
                    size_t source_index, kernel_index;
                    for(size_t row = 0; row < nr_rows_kernel; ++row) {
                        source_index = index(source, first_row_source + row,
                            first_col_source + 0);
                        kernel_index = index(kernel, row, 0);
                        for(size_t col = 0; col < nr_cols_kernel; ++col) {
                            if(std::get<0>(input_no_data_policy).is_no_data(
                                    source_index)) {
                                if(AFNP::value(
                                        input_no_data_policy,
                                        source,
                                        size(source, 0),
                                        size(source, 1),
                                        first_row_source + row,
                                        first_col_source + col,
                                        alternative_value)) {
                                    HandleValue<SV, SW, V, W>::apply(
                                        alternative_value,
                                        get(kernel, kernel_index),
                                        sum_of_values, sum_of_weights);
                                    value_seen = true;
                                }
                            }
                            else {
                                HandleValue<SV, SW, V, W>::apply(
                                    get(source, source_index),
                                    get(kernel, kernel_index),
                                    sum_of_values, sum_of_weights);
                                value_seen = true;
                            }

                            ++source_index;
                            ++kernel_index;
                        }
                    }
                }


                // The result can go out of range when the convolution
                // results in an infinite value. Normalizing such a
                // value makes no sense.

                // TODO OutOfRangePolicy must handle integral results too.
                if(!value_seen || !OORP::within_range(sum_of_values)) {
                    output_no_data_policy.mark_as_no_data(index_);
                }
                else {
                    get(destination, index_) = NP::normalize(sum_of_values,
                        sum_of_weights);
                }

                ++first_col_source;
                ++index_;
            }

            ++first_row_source;
        }
    }

};


template<>
struct ConvolveInnerPart<false>
{

    template<
        typename AlternativeForNoDataPolicy,
        typename NormalizePolicy,
        template<typename, typename> class OutOfRangePolicy,
        typename InputNoDataPolicy,
        typename OutputNoDataPolicy,
        typename SourceImage,
        typename Kernel,
        typename DestinationImage>
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

        using V = value_type<SourceImage>;
        using SV = accumulate_type<V>;
        using W = value_type<Kernel>;
        using SW = accumulate_type<W>;

        SV sum_of_values;
        SW sum_of_weights;
        V alternative_value;

        using OORP = OutOfRangePolicy<value_type<SourceImage>,
            value_type<DestinationImage>>;
        using NP = NormalizePolicy;
        using AFNP = AlternativeForNoDataPolicy;


        // Here, we don't have to multiply the cell values by the kernel
        // weights. Most probably this is because the kernel weights are
        // boolean values.
        // Instead of multiplying cell values by kernel weights, we just
        // need to add them, which is cheaper. Also, because we don't need
        // to add all cells that lie within the kernel window. We
        // only need to add those cells for which the kernel weight
        // evaluates to true.
        // To prevent testing the kernel weights over and over again for
        // true-ness, we determine the offsets in de image, relative
        // to the image cell positioned in the upper left kernel window
        // cell, for which the corresponding kernel weight evaluates to true.
        // Then we just need to iterate over this collection,
        // calculate the real image cell index, and add its value (if not
        // skipped for no-data-ness).

        std::vector<std::tuple<size_t, size_t, size_t>> cell_offset_tuples;
        {
            cell_offset_tuples.reserve(size(kernel));
            size_t nr_cols = size(source, 1);

            for(size_t i = 0, r = 0; r < nr_rows_kernel; ++r) {
                for(size_t c = 0; c < nr_cols_kernel; ++c) {
                    if(get(kernel, i)) {
                        cell_offset_tuples.emplace_back(r, c, r * nr_cols + c);
                    }
                    i++;
                }
            }
        }


        bool value_seen;

        size_t index_;

        // Loop over all cells that are situated in the inner part. The kernel
        // does not extent outside of the source image.
        for(size_t row_source = index_ranges[0].begin(); row_source <
                index_ranges[0].end(); ++row_source) {

            first_col_source = index_ranges[1].begin() - radius_;

            index_ = index(source, row_source, index_ranges[1].begin());

            for(size_t col_source = index_ranges[1].begin(); col_source <
                    index_ranges[1].end(); ++col_source) {

                sum_of_values = AccumulationTraits<V>::zero;
                sum_of_weights = AccumulationTraits<W>::zero;
                value_seen = false;

                if(std::get<0>(input_no_data_policy).is_no_data(index_)) {
                    output_no_data_policy.mark_as_no_data(index_);
                }
                else {
                    // Handle cells positioned in the kernel, and in the source
                    // image.
                    size_t source_index = index(source, first_row_source,
                        first_col_source);
                    size_t row, col, offset;

                    for(auto const& tuple: cell_offset_tuples) {
                        row = get<0>(tuple);
                        col = get<1>(tuple);
                        offset = get<2>(tuple);

                        if(std::get<0>(input_no_data_policy).is_no_data(
                                source_index)) {
                            if(AFNP::value(
                                    input_no_data_policy,
                                    source,
                                    size(source, 0),
                                    size(source, 1),
                                    first_row_source + row,
                                    first_col_source + col,
                                    alternative_value)) {
                                sum_of_values += alternative_value;
                                sum_of_weights += 1;
                                value_seen = true;
                            }
                        }
                        else {
                            sum_of_values += get(source, source_index + offset),
                            sum_of_weights += 1;
                            value_seen = true;
                        }
                    }
                }

                // The result can go out of range when the convolution
                // results in an infinite value. Normalizing such a
                // value makes no sense.

                // TODO OutOfRangePolicy must handle integral results too.
                if(!value_seen || !OORP::within_range(sum_of_values)) {
                    output_no_data_policy.mark_as_no_data(index_);
                }
                else {
                    get(destination, index_) = NP::normalize(sum_of_values,
                        sum_of_weights);
                }

                ++first_col_source;
                ++index_;
            }

            ++first_row_source;
        }
    }

};


template<
    typename AlternativeForNoDataPolicy,
    typename NormalizePolicy,
    typename OutOfImagePolicy,
    template<typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename SourceImage,
    typename Kernel,
    typename DestinationImage,
    typename ExecutionPolicy>
struct Convolve
{
};


template<
    typename AlternativeForNoDataPolicy,
    typename NormalizePolicy,
    typename OutOfImagePolicy,
    template<typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename SourceImage,
    typename Kernel,
    typename DestinationImage>
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
        SequentialExecutionPolicy& /* execution_policy */,
        SourceImage const& source,
        Kernel const& kernel,
        DestinationImage& destination)
    {
        // Corners.
        dispatch::ConvolveNorthWestCorner::
            template apply<AlternativeForNoDataPolicy, NormalizePolicy,
                OutOfImagePolicy, OutOfRangePolicy>(
                    input_no_data_policy, output_no_data_policy,
                    source, kernel, destination);
        dispatch::ConvolveNorthEastCorner::
            template apply<AlternativeForNoDataPolicy, NormalizePolicy,
                OutOfImagePolicy, OutOfRangePolicy>(
                    input_no_data_policy, output_no_data_policy,
                    source, kernel, destination);
        dispatch::ConvolveSouthWestCorner::
            template apply<AlternativeForNoDataPolicy, NormalizePolicy,
                OutOfImagePolicy, OutOfRangePolicy>(
                    input_no_data_policy, output_no_data_policy,
                    source, kernel, destination);
        dispatch::ConvolveSouthEastCorner::
            template apply<AlternativeForNoDataPolicy, NormalizePolicy,
                OutOfImagePolicy, OutOfRangePolicy>(
                    input_no_data_policy, output_no_data_policy,
                    source, kernel, destination);

        // Sides.
        dispatch::ConvolveNorthSide::
            template apply<AlternativeForNoDataPolicy, NormalizePolicy,
                OutOfImagePolicy, OutOfRangePolicy>(
                    input_no_data_policy, output_no_data_policy,
                    source, kernel, destination);
        dispatch::ConvolveWestSide::
            template apply<AlternativeForNoDataPolicy, NormalizePolicy,
                OutOfImagePolicy, OutOfRangePolicy>(
                    input_no_data_policy, output_no_data_policy,
                    source, kernel, destination);
        dispatch::ConvolveEastSide::
            template apply<AlternativeForNoDataPolicy, NormalizePolicy,
                OutOfImagePolicy, OutOfRangePolicy>(
                    input_no_data_policy, output_no_data_policy,
                    source, kernel, destination);
        dispatch::ConvolveSouthSide::
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
                            size(source, 0) - fern::radius(kernel)),
                        IndexRange(fern::radius(kernel),
                            size(source, 1) - fern::radius(kernel))},
                    source, kernel, destination);
    }

};


#define CREATE_BORDER_TASK(                                                    \
    part)                                                                      \
{                                                                              \
    auto function = std::bind((BorderFunction)                                 \
        dispatch::Convolve##part::template                                     \
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
    typename AlternativeForNoDataPolicy,
    typename NormalizePolicy,
    typename OutOfImagePolicy,
    template<typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename SourceImage,
    typename Kernel,
    typename DestinationImage>
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
        ParallelExecutionPolicy& execution_policy,
        SourceImage const& source,
        Kernel const& kernel,
        DestinationImage& destination)
    {
        ThreadPool& pool(execution_policy.thread_pool());
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
        assert(size(source, 0) == size(destination, 0));
        assert(size(source, 1) == size(destination, 1));
        assert(2 * fern::radius(kernel) <= size(source, 0));
        assert(2 * fern::radius(kernel) <= size(source, 1));

        size_t const size1 = size(source, 0) - 2 * fern::radius(kernel);
        size_t const size2 = size(source, 1) - 2 * fern::radius(kernel);
        std::vector<IndexRanges<2>> index_ranges =
            fern::algorithm::index_ranges(pool.size(), size1, size2);

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


template<
    typename AlternativeForNoDataPolicy,
    typename NormalizePolicy,
    typename OutOfImagePolicy,
    template<typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename SourceImage,
    typename Kernel,
    typename DestinationImage>
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
    ExecutionPolicy>
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy& execution_policy,
        SourceImage const& source,
        Kernel const& kernel,
        DestinationImage& destination)
    {
        switch(execution_policy.which()) {
            case fern::algorithm::detail::sequential_execution_policy_id: {
                Convolve<AlternativeForNoDataPolicy, NormalizePolicy,
                    OutOfImagePolicy, OutOfRangePolicy, InputNoDataPolicy,
                    OutputNoDataPolicy, SourceImage, Kernel, DestinationImage,
                    SequentialExecutionPolicy>::apply(
                        input_no_data_policy, output_no_data_policy,
                        boost::get<SequentialExecutionPolicy>(execution_policy),
                        source, kernel, destination);
                break;
            }
            case fern::algorithm::detail::parallel_execution_policy_id: {
                Convolve<AlternativeForNoDataPolicy, NormalizePolicy,
                    OutOfImagePolicy, OutOfRangePolicy, InputNoDataPolicy,
                    OutputNoDataPolicy, SourceImage, Kernel, DestinationImage,
                    ParallelExecutionPolicy>::apply(
                        input_no_data_policy, output_no_data_policy,
                        boost::get<ParallelExecutionPolicy>(execution_policy),
                        source, kernel, destination);
                break;
            }
        }
    }

};

} // namespace dispatch


template<
    typename AlternativeForNoDataPolicy,
    typename NormalizePolicy,
    typename OutOfImagePolicy,
    template<typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename SourceImage,
    typename Kernel,
    typename DestinationImage
>
void convolve(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
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
} // namespace algorithm
} // namespace fern
