// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once


namespace fern {
namespace algorithm {
namespace convolve {

/*!
    @ingroup    fern_algorithm_convolution_group
    @brief      NormalizePolicy which normalizes convoluted values by the
                sum of the kernel weights.
    @sa         @ref fern_algorithm_convolution_policies
*/
class DivideByWeights
{

public:

    template<typename Value, typename Count>
    static Value   normalize           (Value const& value,
                                        Count const& count);

};


template<
    typename Value,
    typename Count
>
inline Value DivideByWeights::normalize(
        Value const& value,
        Count const& count)
{
    return value / count;
}

} // namespace convolve
} // namespace algorithm
} // namespace fern
