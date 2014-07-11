#pragma once


namespace fern {
namespace convolve {

//! NormalizePolicy which normalizes convoluted values by the sum of the kernel weights.
/*!
    \sa            @ref fern_algorithm_convolution_policies
*/
class DivideByWeights
{

public:

    template<class Value, class Count>
    static Value   normalize           (Value const& value,
                                        Count const& count);

};


template<
    class Value,
    class Count
>
inline Value DivideByWeights::normalize(
        Value const& value,
        Count const& count)
{
    return value / count;
}

} // namespace convolve
} // namespace fern
