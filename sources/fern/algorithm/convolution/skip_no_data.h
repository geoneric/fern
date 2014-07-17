#pragma once
#include <cstddef>


namespace fern {
namespace convolve {

//! AlternativeForNoDataPolicy which discards no-data values.
/*!
    \sa            @ref fern_algorithm_convolution_policies
*/
class SkipNoData
{

public:

    template<class InputNoDataPolicy, class SourceImage, class Value>
    static constexpr bool
                   value               (InputNoDataPolicy const&
                                            input_no_data_policy,
                                        SourceImage const& source,
                                        size_t const nr_rows,
                                        size_t const nr_cols,
                                        size_t const row,
                                        size_t const col,
                                        Value& value);

};


template<
    class InputNoDataPolicy,
    class SourceImage,
    class Value
>
inline constexpr bool SkipNoData::value(
        InputNoDataPolicy const& /* input_no_data_policy */,
        SourceImage const& /* source */,
        size_t const /* nr_rows */,
        size_t const /* nr_cols */,
        size_t const /* row */,
        size_t const /* col */,
        Value& /* value */)
{
    return false;
}

} // namespace convolve
} // namespace fern
