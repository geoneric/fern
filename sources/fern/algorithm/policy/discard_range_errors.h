#pragma once


namespace fern {

template<
    class... Parameters>
class DiscardRangeErrors {

public:

    static constexpr bool
                   within_range        (Parameters const&... parameters);

};


template<
    class... Parameters>
inline constexpr bool DiscardRangeErrors<Parameters...>::within_range(
    Parameters const&... /* parameters */)
{
    return true;
}


namespace binary {

template<
    class Value1,
    class Value2,
    class Result>
using DiscardRangeErrors = DiscardRangeErrors<Value1, Value2, Result>;

} // namespace binary
} // namespace fern
