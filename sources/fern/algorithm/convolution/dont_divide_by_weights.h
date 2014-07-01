#pragma once


namespace fern {
namespace convolve {

class DontDivideByWeights
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
inline Value DontDivideByWeights::normalize(
        Value const& value,
        Count const& /* count */)
{
    return value;
}

} // namespace convolve
} // namespace fern
