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

    template<class InputNoDataPolicy, class SourceImage, class Value>
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

    template<class InputNoDataPolicy, class SourceImage, class Value>
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

    template<class InputNoDataPolicy, class SourceImage, class Value>
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

    template<class InputNoDataPolicy, class SourceImage, class Value>
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

    template<class InputNoDataPolicy, class SourceImage, class Value>
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

    template<class InputNoDataPolicy, class SourceImage, class Value>
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

    template<class InputNoDataPolicy, class SourceImage, class Value>
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

    template<class InputNoDataPolicy, class SourceImage, class Value>
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
    class InputNoDataPolicy,
    class SourceImage,
    class Value
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
    class InputNoDataPolicy,
    class SourceImage,
    class Value
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
    class InputNoDataPolicy,
    class SourceImage,
    class Value
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
    class InputNoDataPolicy,
    class SourceImage,
    class Value
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
    class InputNoDataPolicy,
    class SourceImage,
    class Value
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
    class InputNoDataPolicy,
    class SourceImage,
    class Value
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
    class InputNoDataPolicy,
    class SourceImage,
    class Value
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
    class InputNoDataPolicy,
    class SourceImage,
    class Value
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
