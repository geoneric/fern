#pragma once
#include <algorithm>
#include <cassert>


namespace fern {
namespace convolve {

//! OutOfImagePolicy which calculates focal average for out-of-image values.
/*!
    \sa            @ref fern_algorithm_convolution_policies
*/
class ReplaceOutOfImageByFocalAverage
{

public:

    template<class InputNoDataPolicy, class SourceImage, class Value>
    static bool    value_north_west    (InputNoDataPolicy const&
                                            input_no_data_policy,
                                        SourceImage const& source,
                                        size_t const out_of_image_kernel_row,
                                        size_t const out_of_image_kernel_col,
                                        size_t const first_row_kernel,
                                        size_t const first_col_kernel,
                                        size_t const nr_rows_kernel,
                                        size_t const nr_cols_kernel,
                                        size_t const first_row_source,
                                        size_t const first_col_source,
                                        Value& value);


    template<class InputNoDataPolicy, class SourceImage, class Value>
    static bool    value_north_east    (InputNoDataPolicy const&
                                            input_no_data_policy,
                                        SourceImage const& source,
                                        size_t const out_of_image_kernel_row,
                                        size_t const out_of_image_kernel_col,
                                        size_t const first_row_kernel,
                                        size_t const first_col_kernel,
                                        size_t const nr_rows_kernel,
                                        size_t const nr_cols_kernel,
                                        size_t const first_row_source,
                                        size_t const first_col_source,
                                        Value& value);


    template<class InputNoDataPolicy, class SourceImage, class Value>
    static bool    value_south_west    (InputNoDataPolicy const&
                                            input_no_data_policy,
                                        SourceImage const& source,
                                        size_t const out_of_image_kernel_row,
                                        size_t const out_of_image_kernel_col,
                                        size_t const first_row_kernel,
                                        size_t const first_col_kernel,
                                        size_t const nr_rows_kernel,
                                        size_t const nr_cols_kernel,
                                        size_t const first_row_source,
                                        size_t const first_col_source,
                                        Value& value);

    template<class InputNoDataPolicy, class SourceImage, class Value>
    static bool    value_south_east    (InputNoDataPolicy const&
                                            input_no_data_policy,
                                        SourceImage const& source,
                                        size_t const out_of_image_kernel_row,
                                        size_t const out_of_image_kernel_col,
                                        size_t const first_row_kernel,
                                        size_t const first_col_kernel,
                                        size_t const nr_rows_kernel,
                                        size_t const nr_cols_kernel,
                                        size_t const first_row_source,
                                        size_t const first_col_source,
                                        Value& value);

    template<class InputNoDataPolicy, class SourceImage, class Value>
    static bool    value_north         (InputNoDataPolicy const&
                                            input_no_data_policy,
                                        SourceImage const& source,
                                        size_t const out_of_image_kernel_row,
                                        size_t const out_of_image_kernel_col,
                                        size_t const first_row_kernel,
                                        size_t const first_col_kernel,
                                        size_t const nr_rows_kernel,
                                        size_t const nr_cols_kernel,
                                        size_t const first_row_source,
                                        size_t const first_col_source,
                                        Value& value);

    template<class InputNoDataPolicy, class SourceImage, class Value>
    static bool    value_west          (InputNoDataPolicy const&
                                            input_no_data_policy,
                                        SourceImage const& source,
                                        size_t const out_of_image_kernel_row,
                                        size_t const out_of_image_kernel_col,
                                        size_t const first_row_kernel,
                                        size_t const first_col_kernel,
                                        size_t const nr_rows_kernel,
                                        size_t const nr_cols_kernel,
                                        size_t const first_row_source,
                                        size_t const first_col_source,
                                        Value& value);

    template<class InputNoDataPolicy, class SourceImage, class Value>
    static bool    value_east          (InputNoDataPolicy const&
                                            input_no_data_policy,
                                        SourceImage const& source,
                                        size_t const out_of_image_kernel_row,
                                        size_t const out_of_image_kernel_col,
                                        size_t const first_row_kernel,
                                        size_t const first_col_kernel,
                                        size_t const nr_rows_kernel,
                                        size_t const nr_cols_kernel,
                                        size_t const first_row_source,
                                        size_t const first_col_source,
                                        Value& value);

    template<class InputNoDataPolicy, class SourceImage, class Value>
    static bool    value_south         (InputNoDataPolicy const&
                                            input_no_data_policy,
                                        SourceImage const& source,
                                        size_t const out_of_image_kernel_row,
                                        size_t const out_of_image_kernel_col,
                                        size_t const first_row_kernel,
                                        size_t const first_col_kernel,
                                        size_t const nr_rows_kernel,
                                        size_t const nr_cols_kernel,
                                        size_t const first_row_source,
                                        size_t const first_col_source,
                                        Value& value);

};


template<
    class InputNoDataPolicy,
    class SourceImage,
    class Value
>
inline bool ReplaceOutOfImageByFocalAverage::value_north_west(
        InputNoDataPolicy const& input_no_data_policy,
        SourceImage const& source,
        size_t const out_of_image_kernel_row,
        size_t const out_of_image_kernel_col,
        size_t const first_row_kernel,
        size_t const first_col_kernel,
        size_t const /* nr_rows_kernel */,
        size_t const /* nr_cols_kernel */,
        size_t const first_row_source,
        size_t const first_col_source,
        Value& value)
{
    Value sum_of_values{0};
    size_t nr_values_seen{0};
    size_t const radius{1};

    if(out_of_image_kernel_row < first_row_kernel) {
        // Row north of image.

        if(out_of_image_kernel_col < first_col_kernel) {
            // Column west of image.

            // Check if our neighborhood extents into the source image.
            if(first_row_kernel - out_of_image_kernel_row <= radius &&
                    first_col_kernel - out_of_image_kernel_col <= radius) {

                size_t row = first_row_source;
                size_t const last_row = radius - (first_row_kernel -
                    out_of_image_kernel_row);

                for(; row <= last_row; ++row) {

                    size_t col = first_col_source;
                    size_t const last_col = radius - (first_col_kernel -
                        out_of_image_kernel_col);

                    for(; col <= last_col; ++col) {

                        if(!input_no_data_policy.is_no_data(row, col)) {
                            sum_of_values += get(source, row, col);
                            ++nr_values_seen;
                        }
                    }
                }
            }
        }
        else {
            // Column overlaps with image.

            // Check if our neighborhood extents into the source image.
            if(first_row_kernel - out_of_image_kernel_row <= radius) {

                size_t row = first_row_source;
                size_t const last_row = radius - (first_row_kernel -
                    out_of_image_kernel_row);

                for(; row <= last_row; ++row) {

                    size_t col = out_of_image_kernel_col - first_col_kernel >=
                        radius ? out_of_image_kernel_col - first_col_kernel -
                        radius : 0;
                    size_t const last_col = out_of_image_kernel_col -
                        first_col_kernel + radius;

                    for(; col <= last_col; ++col) {

                        if(!input_no_data_policy.is_no_data(row, col)) {
                            sum_of_values += get(source, row, col);
                            ++nr_values_seen;
                        }
                    }
                }
            }
        }
    }
    else {
        // Row overlaps with image.
        // Column west of image.

        assert(out_of_image_kernel_col < first_col_kernel);

        // Check if our neighborhood extents into the source image.
        if(first_col_kernel - out_of_image_kernel_col <= radius) {
            size_t row = out_of_image_kernel_row - first_row_kernel >= radius
                ? out_of_image_kernel_row -first_row_kernel - radius
                : 0;
            size_t const last_row = out_of_image_kernel_row -
                first_row_kernel + radius;

            for(; row <= last_row; ++row) {
                size_t col = first_col_source;
                size_t const last_col = radius - (first_col_kernel -
                    out_of_image_kernel_col);

                for(; col <= last_col; ++col) {
                    if(!input_no_data_policy.is_no_data(row, col)) {
                        sum_of_values += get(source, row, col);
                        ++nr_values_seen;
                    }
                }
            }
        }
    }

    if(nr_values_seen > 0) {
        value = sum_of_values / nr_values_seen;
    }

    return nr_values_seen > 0;
}


template<
    class InputNoDataPolicy,
    class SourceImage,
    class Value
>
inline bool ReplaceOutOfImageByFocalAverage::value_north_east(
        InputNoDataPolicy const& input_no_data_policy,
        SourceImage const& source,
        size_t const out_of_image_kernel_row,
        size_t const out_of_image_kernel_col,
        size_t const first_row_kernel,
        size_t const first_col_kernel,
        size_t const /* nr_rows_kernel */,
        size_t const nr_cols_kernel,
        size_t const first_row_source,
        size_t const first_col_source,
        Value& value)
{
    Value sum_of_values{0};
    size_t nr_values_seen{0};
    size_t const radius{1};

    size_t const last_col_kernel = first_col_kernel + nr_cols_kernel - 1;
    size_t const last_col_source = size(source, 1) - 1;

    if(out_of_image_kernel_row < first_row_kernel) {
        // Row north of image.

        if(out_of_image_kernel_col > last_col_kernel) {
            // Column east of image.

            // Check if our neighborhood extents into the source image.
            if(first_row_kernel - out_of_image_kernel_row <= radius &&
                    out_of_image_kernel_col - last_col_kernel <= radius) {

                size_t row = first_row_source;
                size_t const last_row = radius - (first_row_kernel -
                    out_of_image_kernel_row);

                for(; row <= last_row; ++row) {

                    size_t col = last_col_source + (out_of_image_kernel_col -
                        last_col_kernel - radius);
                    size_t const last_col = last_col_source;

                    for(; col <= last_col; ++col) {

                        if(!input_no_data_policy.is_no_data(row, col)) {
                            sum_of_values += get(source, row, col);
                            ++nr_values_seen;
                        }
                    }
                }
            }
        }
        else {
            // Column overlaps with image.

            // Check if our neighborhood extents into the source image.
            if(first_row_kernel - out_of_image_kernel_row <= radius) {

                size_t row = first_row_source;
                size_t const last_row = radius - (first_row_kernel -
                    out_of_image_kernel_row);

                size_t const col_source = first_col_source +
                    out_of_image_kernel_col;

                for(; row <= last_row; ++row) {

                    size_t col = col_source - radius;
                    size_t last_col = std::min(col_source + radius,
                        last_col_source);

                    for(; col <= last_col; ++col) {

                        if(!input_no_data_policy.is_no_data(row, col)) {
                            sum_of_values += get(source, row, col);
                            ++nr_values_seen;
                        }
                    }
                }
            }
        }
    }
    else {
        // Row overlaps with image.
        // Column east of image.

        assert(out_of_image_kernel_col > last_col_kernel);

        if(out_of_image_kernel_col - last_col_kernel <= radius) {

            size_t row =
                out_of_image_kernel_row - first_row_kernel >= radius
                    ? out_of_image_kernel_row - first_row_kernel - radius
                    : 0;

            size_t const last_row = out_of_image_kernel_row -
                first_row_kernel + radius;

            for(; row <= last_row; ++row) {

                size_t col = last_col_source + (out_of_image_kernel_col -
                    last_col_kernel - radius);
                size_t const last_col = last_col_source;

                for(; col <= last_col; ++col) {

                    if(!input_no_data_policy.is_no_data(row, col)) {
                        sum_of_values += get(source, row, col);
                        ++nr_values_seen;
                    }
                }
            }
        }
    }

    if(nr_values_seen > 0) {
        value = sum_of_values / nr_values_seen;
    }

    return nr_values_seen > 0;
}


template<
    class InputNoDataPolicy,
    class SourceImage,
    class Value
>
inline bool ReplaceOutOfImageByFocalAverage::value_south_west(
        InputNoDataPolicy const& input_no_data_policy,
        SourceImage const& source,
        size_t const out_of_image_kernel_row,
        size_t const out_of_image_kernel_col,
        size_t const first_row_kernel,
        size_t const first_col_kernel,
        size_t const nr_rows_kernel,
        size_t const /* nr_cols_kernel */,
        size_t const first_row_source,
        size_t const first_col_source,
        Value& value)
{
    Value sum_of_values{0};
    size_t nr_values_seen{0};
    size_t const radius{1};

    size_t const last_row_kernel = first_row_kernel + nr_rows_kernel - 1;
    size_t const last_row_source = size(source, 0) - 1;

    if(out_of_image_kernel_row > last_row_kernel) {
        // Row south of image.

        if(out_of_image_kernel_col < first_col_kernel) {
            // Column west of image.

            // Check if our neighborhood extents into the source image.
            if(out_of_image_kernel_row - last_row_kernel <= radius &&
                    first_col_kernel - out_of_image_kernel_col <= radius) {

                size_t row = last_row_source + (out_of_image_kernel_row -
                    last_row_kernel - radius);
                size_t const last_row = last_row_source;

                for(; row <= last_row; ++row) {

                    size_t col = first_col_source;
                    size_t const last_col = radius - (first_col_kernel -
                        out_of_image_kernel_col);

                    for(; col <= last_col; ++col) {

                        if(!input_no_data_policy.is_no_data(row, col)) {
                            sum_of_values += get(source, row, col);
                            ++nr_values_seen;
                        }
                    }
                }
            }
        }
        else {
            // Column overlaps with image.

            // Check if our neighborhood extents into the source image.
            if(out_of_image_kernel_row - last_row_kernel <= radius) {

                size_t row = last_row_source + (out_of_image_kernel_row -
                    last_row_kernel - radius);
                size_t const last_row = last_row_source;

                for(; row <= last_row; ++row) {

                    size_t col = out_of_image_kernel_col - first_col_kernel >=
                        radius ? out_of_image_kernel_col - first_col_kernel -
                        radius : 0;
                    size_t const last_col = out_of_image_kernel_col -
                        first_col_kernel + radius;

                    for(; col <= last_col; ++col) {

                        if(!input_no_data_policy.is_no_data(row, col)) {
                            sum_of_values += get(source, row, col);
                            ++nr_values_seen;
                        }
                    }
                }
            }
        }
    }
    else {
        // Row overlaps with image.
        // Column west of image.

        assert(out_of_image_kernel_col < first_col_kernel);

        // Check if our neighborhood extents into the source image.
        if(first_col_kernel - out_of_image_kernel_col <= radius) {

            size_t const row_source = first_row_source +
                out_of_image_kernel_row;
            size_t row = row_source - radius;
            size_t last_row = std::min(row_source + radius, last_row_source);

            for(; row <= last_row; ++row) {

                size_t col = first_col_source;
                size_t const last_col = radius - (first_col_kernel -
                    out_of_image_kernel_col);

                for(; col <= last_col; ++col) {

                    if(!input_no_data_policy.is_no_data(row, col)) {
                        sum_of_values += get(source, row, col);
                        ++nr_values_seen;
                    }
                }
            }
        }
    }


    if(nr_values_seen > 0) {
        value = sum_of_values / nr_values_seen;
    }

    return nr_values_seen > 0;
}


template<
    class InputNoDataPolicy,
    class SourceImage,
    class Value
>
inline bool ReplaceOutOfImageByFocalAverage::value_south_east(
        InputNoDataPolicy const& input_no_data_policy,
        SourceImage const& source,
        size_t const out_of_image_kernel_row,
        size_t const out_of_image_kernel_col,
        size_t const first_row_kernel,
        size_t const first_col_kernel,
        size_t const nr_rows_kernel,
        size_t const nr_cols_kernel,
        size_t const first_row_source,
        size_t const first_col_source,
        Value& value)
{
    Value sum_of_values{0};
    size_t nr_values_seen{0};
    size_t const radius{1};

    size_t const last_row_kernel = first_row_kernel + nr_rows_kernel - 1;
    size_t const last_row_source = size(source, 0) - 1;
    size_t const last_col_kernel = first_col_kernel + nr_cols_kernel - 1;
    size_t const last_col_source = size(source, 1) - 1;

    if(out_of_image_kernel_row > last_row_kernel) {
        // Row south of image.

        if(out_of_image_kernel_col > last_col_kernel) {
            // Column east of image.

            // Check if our neighborhood extents into the source image.
            if(out_of_image_kernel_row - last_row_kernel <= radius &&
                    out_of_image_kernel_col - last_col_kernel <= radius) {

                size_t row = last_row_source + (out_of_image_kernel_row -
                    last_row_kernel - radius);
                size_t const last_row = last_row_source;

                for(; row <= last_row; ++row) {

                    size_t col = last_col_source + (out_of_image_kernel_col -
                        last_col_kernel - radius);
                    size_t const last_col = last_col_source;

                    for(; col <= last_col; ++col) {

                        if(!input_no_data_policy.is_no_data(row, col)) {
                            sum_of_values += get(source, row, col);
                            ++nr_values_seen;
                        }
                    }
                }
            }
        }
        else {
            // Column overlaps with image.

            // Check if our neighborhood extents into the source image.
            if(out_of_image_kernel_row - last_row_kernel <= radius) {

                size_t row = last_row_source + (out_of_image_kernel_row -
                    last_row_kernel - radius);
                size_t const last_row = last_row_source;

                size_t const col_source = first_col_source +
                    out_of_image_kernel_col;

                for(; row <= last_row; ++row) {

                    size_t col = col_source - radius;
                    size_t last_col = std::min(col_source + radius,
                        last_col_source);

                    for(; col <= last_col; ++col) {

                        if(!input_no_data_policy.is_no_data(row, col)) {
                            sum_of_values += get(source, row, col);
                            ++nr_values_seen;
                        }
                    }
                }
            }
        }
    }
    else {
        // Row overlaps with image.
        // Column west of image.

        assert(out_of_image_kernel_col > last_col_kernel);

        // Check if our neighborhood extents into the source image.
        if(out_of_image_kernel_col - last_col_kernel <= radius) {

            size_t const row_source = first_row_source +
                out_of_image_kernel_row;
            size_t row = row_source - radius;
            size_t last_row = std::min(row_source + radius, last_row_source);

            for(; row <= last_row; ++row) {

                size_t col = last_col_source + (out_of_image_kernel_col -
                    last_col_kernel - radius);
                size_t const last_col = last_col_source;

                for(; col <= last_col; ++col) {
                    if(!input_no_data_policy.is_no_data(row, col)) {
                        sum_of_values += get(source, row, col);
                        ++nr_values_seen;
                    }
                }
            }
        }
    }


    if(nr_values_seen > 0) {
        value = sum_of_values / nr_values_seen;
    }

    return nr_values_seen > 0;
}


template<
    class InputNoDataPolicy,
    class SourceImage,
    class Value
>
inline bool ReplaceOutOfImageByFocalAverage::value_north(
        InputNoDataPolicy const& input_no_data_policy,
        SourceImage const& source,
        size_t const out_of_image_kernel_row,
        size_t const out_of_image_kernel_col,
        size_t const first_row_kernel,
        size_t const /* first_col_kernel */,
        size_t const /* nr_rows_kernel */,
        size_t const /* nr_cols_kernel */,
        size_t const first_row_source,
        size_t const first_col_source,
        Value& value)
{
    Value sum_of_values{0};
    size_t nr_values_seen{0};
    size_t const radius{1};

    size_t const last_col_source = size(source, 1) - 1;

    assert(out_of_image_kernel_row < first_row_kernel);

    // Check if our neighborhood extents into the source image.
    if(first_row_kernel - out_of_image_kernel_row <= radius) {

        size_t row = first_row_source;
        size_t const last_row = radius - (first_row_kernel -
            out_of_image_kernel_row);

        size_t const col_source = first_col_source +
            out_of_image_kernel_col;

        for(; row <= last_row; ++row) {

            size_t col = radius > col_source ? 0 : col_source - radius;
            size_t last_col = std::min(col_source + radius,
                last_col_source);

            for(; col <= last_col; ++col) {

                if(!input_no_data_policy.is_no_data(row, col)) {
                    sum_of_values += get(source, row, col);
                    ++nr_values_seen;
                }
            }
        }
    }

    if(nr_values_seen > 0) {
        value = sum_of_values / nr_values_seen;
    }

    return nr_values_seen > 0;
}


template<
    class InputNoDataPolicy,
    class SourceImage,
    class Value
>
inline bool ReplaceOutOfImageByFocalAverage::value_west(
        InputNoDataPolicy const& input_no_data_policy,
        SourceImage const& source,
        size_t const out_of_image_kernel_row,
        size_t const out_of_image_kernel_col,
        size_t const /* first_row_kernel */,
        size_t const first_col_kernel,
        size_t const /* nr_rows_kernel */,
        size_t const /* nr_cols_kernel */,
        size_t const first_row_source,
        size_t const first_col_source,
        Value& value)
{
    Value sum_of_values{0};
    size_t nr_values_seen{0};
    size_t const radius{1};

    size_t const last_row_source = size(source, 0) - 1;

    assert(out_of_image_kernel_col < first_col_kernel);

    size_t const row_source = first_row_source + out_of_image_kernel_row;

    // Check if our neighborhood extents into the source image.
    if(first_col_kernel - out_of_image_kernel_col <= radius) {

        size_t row = radius > row_source ? 0 : row_source - radius;
        size_t last_row = std::min(row_source + radius, last_row_source);

        for(; row <= last_row; ++row) {

            size_t col = first_col_source;
            size_t const last_col = radius - (first_col_kernel -
                out_of_image_kernel_col);

            for(; col <= last_col; ++col) {

                if(!input_no_data_policy.is_no_data(row, col)) {
                    sum_of_values += get(source, row, col);
                    ++nr_values_seen;
                }
            }
        }
    }

    if(nr_values_seen > 0) {
        value = sum_of_values / nr_values_seen;
    }

    return nr_values_seen > 0;
}


template<
    class InputNoDataPolicy,
    class SourceImage,
    class Value
>
inline bool ReplaceOutOfImageByFocalAverage::value_east(
        InputNoDataPolicy const& input_no_data_policy,
        SourceImage const& source,
        size_t const out_of_image_kernel_row,
        size_t const out_of_image_kernel_col,
        size_t const /* first_row_kernel */,
        size_t const first_col_kernel,
        size_t const /* nr_rows_kernel */,
        size_t const nr_cols_kernel,
        size_t const first_row_source,
        size_t const /* first_col_source */,
        Value& value)
{
    Value sum_of_values{0};
    size_t nr_values_seen{0};
    size_t const radius{1};

    size_t const last_row_source = size(source, 0) - 1;
    size_t const last_col_kernel = first_col_kernel + nr_cols_kernel - 1;
    size_t const last_col_source = size(source, 1) - 1;

    // assert(out_of_image_kernel_row > first_row_kernel);
    assert(out_of_image_kernel_col > last_col_kernel);

    size_t const row_source = first_row_source + out_of_image_kernel_row;

    // Check if our neighborhood extents into the source image.
    if(out_of_image_kernel_col - last_col_kernel <= radius) {

        size_t row = radius > row_source ? 0 : row_source - radius;
        size_t last_row = std::min(row_source + radius, last_row_source);

        for(; row <= last_row; ++row) {

            size_t col = last_col_source + (out_of_image_kernel_col -
                last_col_kernel - radius);
            size_t const last_col = last_col_source;

            for(; col <= last_col; ++col) {

                if(!input_no_data_policy.is_no_data(row, col)) {
                    sum_of_values += get(source, row, col);
                    ++nr_values_seen;
                }
            }
        }
    }

    if(nr_values_seen > 0) {
        value = sum_of_values / nr_values_seen;
    }

    return nr_values_seen > 0;
}


template<
    class InputNoDataPolicy,
    class SourceImage,
    class Value
>
inline bool ReplaceOutOfImageByFocalAverage::value_south(
        InputNoDataPolicy const& input_no_data_policy,
        SourceImage const& source,
        size_t const out_of_image_kernel_row,
        size_t const out_of_image_kernel_col,
        size_t const first_row_kernel,
        size_t const /* first_col_kernel */,
        size_t const nr_rows_kernel,
        size_t const /* nr_cols_kernel */,
        size_t const /* first_row_source */,
        size_t const first_col_source,
        Value& value)
{
    Value sum_of_values{0};
    size_t nr_values_seen{0};
    size_t const radius{1};

    size_t const last_row_kernel = first_row_kernel + nr_rows_kernel - 1;
    size_t const last_row_source = size(source, 0) - 1;
    size_t const last_col_source = size(source, 1) - 1;

    assert(out_of_image_kernel_row > last_row_kernel);

    // Check if our neighborhood extents into the source image.
    if(out_of_image_kernel_row - last_row_kernel <= radius) {

        size_t row = last_row_source + (out_of_image_kernel_row -
            last_row_kernel - radius);
        size_t const last_row = last_row_source;

        size_t const col_source = first_col_source +
            out_of_image_kernel_col;

        for(; row <= last_row; ++row) {

            size_t col = radius > col_source ? 0 : col_source - radius;
            size_t last_col = std::min(col_source + radius,
                last_col_source);

            for(; col <= last_col; ++col) {

                if(!input_no_data_policy.is_no_data(row, col)) {
                    sum_of_values += get(source, row, col);
                    ++nr_values_seen;
                }
            }
        }
    }

    if(nr_values_seen > 0) {
        value = sum_of_values / nr_values_seen;
    }

    return nr_values_seen > 0;
}

} // namespace convolve
} // namespace fern
