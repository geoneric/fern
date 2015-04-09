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
    @brief      AlternativeForNoDataPolicy which discards no-data values.
    @sa         @ref fern_algorithm_convolution_policies
*/
class SkipNoData
{

public:

    template<typename InputNoDataPolicy, typename SourceImage, typename Value>
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
    typename InputNoDataPolicy,
    typename SourceImage,
    typename Value
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
} // namespace algorithm
} // namespace fern
