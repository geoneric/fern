#pragma once


namespace fern {
namespace algorithm {
namespace convolve {

/*!
    @ingroup    fern_algorithm_convolution_group
    @brief      NormalizePolicy which does not normalize convoluted values.
    @sa         @ref fern_algorithm_convolution_policies
*/
class DontDivideByWeights
{

public:

    template<typename Value, typename Count>
    static Value   normalize           (Value const& value,
                                        Count const& count);

};


template<
    typename Value,
    typename Count>
inline Value DontDivideByWeights::normalize(
        Value const& value,
        Count const& /* count */)
{
    return value;
}

} // namespace convolve
} // namespace algorithm
} // namespace fern
