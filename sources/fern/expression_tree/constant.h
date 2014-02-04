#pragma once


namespace fern {

template<
    class Result>
struct Constant
{
    Constant()
        : value()
    {
    }

    Constant(
        Result const& value)
        : value(value)
    {
    }

    typedef Result value_type;

    typedef Constant<Result> result_type;

    Result value;
};

} // namespace fern
