// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <cstddef>


namespace fern {
namespace algorithm {
namespace convolve {

/*!
    @ingroup    fern_algorithm_convolution_group
    @brief      OutOfImagePolicy which treats out-of-image values as no-data.
    @sa         @ref fern_algorithm_convolution_policies
*/
class SkipOutOfImage
{

public:

    template<typename InputNoDataPolicy, typename SourceImage, typename Value>
    static constexpr bool
                   value_north_west    (InputNoDataPolicy const&
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

    template<typename InputNoDataPolicy, typename SourceImage, typename Value>
    static constexpr bool
                   value_north_east    (InputNoDataPolicy const&
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

    template<typename InputNoDataPolicy, typename SourceImage, typename Value>
    static constexpr bool
                   value_south_west    (InputNoDataPolicy const&
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

    template<typename InputNoDataPolicy, typename SourceImage, typename Value>
    static constexpr bool
                   value_south_east    (InputNoDataPolicy const&
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

    template<typename InputNoDataPolicy, typename SourceImage, typename Value>
    static constexpr bool
                   value_north         (InputNoDataPolicy const&
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

    template<typename InputNoDataPolicy, typename SourceImage, typename Value>
    static constexpr bool
                   value_west          (InputNoDataPolicy const&
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

    template<typename InputNoDataPolicy, typename SourceImage, typename Value>
    static constexpr bool
                   value_east          (InputNoDataPolicy const&
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

    template<typename InputNoDataPolicy, typename SourceImage, typename Value>
    static constexpr bool
                   value_south         (InputNoDataPolicy const&
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
    typename InputNoDataPolicy,
    typename SourceImage,
    typename Value
>
inline constexpr bool SkipOutOfImage::value_north_west(
        InputNoDataPolicy const& /* input_no_data_policy */,
        SourceImage const& /* source */,
        size_t const /* out_of_image_kernel_row */,
        size_t const /* out_of_image_kernel_col */,
        size_t const /* first_row_kernel */,
        size_t const /* first_col_kernel */,
        size_t const /* nr_rows_kernel */,
        size_t const /* nr_cols_kernel */,
        size_t const /* first_row_source */,
        size_t const /* first_col_source */,
        Value& /* value */)
{
    return false;
}


template<
    typename InputNoDataPolicy,
    typename SourceImage,
    typename Value
>
inline constexpr bool SkipOutOfImage::value_north_east(
        InputNoDataPolicy const& /* input_no_data_policy */,
        SourceImage const& /* source */,
        size_t const /* out_of_image_kernel_row */,
        size_t const /* out_of_image_kernel_col */,
        size_t const /* first_row_kernel */,
        size_t const /* first_col_kernel */,
        size_t const /* nr_rows_kernel */,
        size_t const /* nr_cols_kernel */,
        size_t const /* first_row_source */,
        size_t const /* first_col_source */,
        Value& /* value */)
{
    return false;
}


template<
    typename InputNoDataPolicy,
    typename SourceImage,
    typename Value
>
inline constexpr bool SkipOutOfImage::value_south_west(
        InputNoDataPolicy const& /* input_no_data_policy */,
        SourceImage const& /* source */,
        size_t const /* out_of_image_kernel_row */,
        size_t const /* out_of_image_kernel_col */,
        size_t const /* first_row_kernel */,
        size_t const /* first_col_kernel */,
        size_t const /* nr_rows_kernel */,
        size_t const /* nr_cols_kernel */,
        size_t const /* first_row_source */,
        size_t const /* first_col_source */,
        Value& /* value */)
{
    return false;
}


template<
    typename InputNoDataPolicy,
    typename SourceImage,
    typename Value
>
inline constexpr bool SkipOutOfImage::value_south_east(
        InputNoDataPolicy const& /* input_no_data_policy */,
        SourceImage const& /* source */,
        size_t const /* out_of_image_kernel_row */,
        size_t const /* out_of_image_kernel_col */,
        size_t const /* first_row_kernel */,
        size_t const /* first_col_kernel */,
        size_t const /* nr_rows_kernel */,
        size_t const /* nr_cols_kernel */,
        size_t const /* first_row_source */,
        size_t const /* first_col_source */,
        Value& /* value */)
{
    return false;
}


template<
    typename InputNoDataPolicy,
    typename SourceImage,
    typename Value
>
inline constexpr bool SkipOutOfImage::value_north(
        InputNoDataPolicy const& /* input_no_data_policy */,
        SourceImage const& /* source */,
        size_t const /* out_of_image_kernel_row */,
        size_t const /* out_of_image_kernel_col */,
        size_t const /* first_row_kernel */,
        size_t const /* first_col_kernel */,
        size_t const /* nr_rows_kernel */,
        size_t const /* nr_cols_kernel */,
        size_t const /* first_row_source */,
        size_t const /* first_col_source */,
        Value& /* value */)
{
    return false;
}


template<
    typename InputNoDataPolicy,
    typename SourceImage,
    typename Value
>
inline constexpr bool SkipOutOfImage::value_west(
        InputNoDataPolicy const& /* input_no_data_policy */,
        SourceImage const& /* source */,
        size_t const /* out_of_image_kernel_row */,
        size_t const /* out_of_image_kernel_col */,
        size_t const /* first_row_kernel */,
        size_t const /* first_col_kernel */,
        size_t const /* nr_rows_kernel */,
        size_t const /* nr_cols_kernel */,
        size_t const /* first_row_source */,
        size_t const /* first_col_source */,
        Value& /* value */)
{
    return false;
}


template<
    typename InputNoDataPolicy,
    typename SourceImage,
    typename Value
>
inline constexpr bool SkipOutOfImage::value_east(
        InputNoDataPolicy const& /* input_no_data_policy */,
        SourceImage const& /* source */,
        size_t const /* out_of_image_kernel_row */,
        size_t const /* out_of_image_kernel_col */,
        size_t const /* first_row_kernel */,
        size_t const /* first_col_kernel */,
        size_t const /* nr_rows_kernel */,
        size_t const /* nr_cols_kernel */,
        size_t const /* first_row_source */,
        size_t const /* first_col_source */,
        Value& /* value */)
{
    return false;
}


template<
    typename InputNoDataPolicy,
    typename SourceImage,
    typename Value
>
inline constexpr bool SkipOutOfImage::value_south(
        InputNoDataPolicy const& /* input_no_data_policy */,
        SourceImage const& /* source */,
        size_t const /* out_of_image_kernel_row */,
        size_t const /* out_of_image_kernel_col */,
        size_t const /* first_row_kernel */,
        size_t const /* first_col_kernel */,
        size_t const /* nr_rows_kernel */,
        size_t const /* nr_cols_kernel */,
        size_t const /* first_row_source */,
        size_t const /* first_col_source */,
        Value& /* value */)
{
    return false;
}

} // namespace convolve
} // namespace algorithm
} // namespace fern
