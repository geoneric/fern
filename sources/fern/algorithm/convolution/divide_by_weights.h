#pragma once


namespace fern {
namespace convolve {

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
