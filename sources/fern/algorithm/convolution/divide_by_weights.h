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
