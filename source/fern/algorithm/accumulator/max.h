#pragma once
#include <algorithm>
#include <limits>


namespace fern {
namespace algorithm {
namespace accumulator {

/*!
    @ingroup    fern_algorithm_accumulator_group
    @brief      Accumulator that calculates the maximum of the added values.
*/
template<
    typename Argument,
    typename Result=Argument>
class Max
{

public:

    /*!
        @brief      During the addition of values, this accumulator's
                    layered maximum value will never go out of range. It
                    is of type @a Argument.
    */
    static bool const out_of_range_risk{false};

                   Max                 ();

    explicit       Max                 (Argument const& value);

    void           operator=           (Argument const& value);

    void           operator()          (Argument const& value);

    Result         operator()          () const;

    Max&           operator|=          (Max const& other);

private:

    Argument       _max;

};


/*!
    @brief      Default construct an instance.

    The layered maximum value is initialized with
    std::numeric_limits<Argument>::lowest().
*/
template<
    typename Argument,
    typename Result>
inline Max<Argument, Result>::Max()

    : _max(std::numeric_limits<Argument>::lowest())

{
}


/*!
    @brief      Construct an instance.

    The layered maximum value is initialized with @a value.
*/
template<
    typename Argument,
    typename Result>
inline Max<Argument, Result>::Max(
    Argument const& value)

    : _max(value)

{
}


/*!
    @brief      Assign @a value to the instance.

    This initializes the instance to be reused again.
*/
template<
    typename Argument,
    typename Result>
inline void Max<Argument, Result>::operator=(
    Argument const& value)
{
    _max = value;
}


/*!
    @brief      Add @a value to the instance.

    In case @a value is larger than the layered maximum value, it is assigned
    to the layered maximum value. Otherwise, the layered maximum value is
    unchanged.
*/
template<
    typename Argument,
    typename Result>
inline void Max<Argument, Result>::operator()(
    Argument const& value)
{
    _max = std::max(_max, value);
}


/*!
    @brief      Return the maximum value seen until now.

    The layered maximum value is cast to the @a Result type.
*/
template<
    typename Argument,
    typename Result>
inline Result Max<Argument, Result>::operator()() const
{
    return static_cast<Result>(_max);
}


/*!
    @brief      Merge @a other with *this.

    The resulting accumulator represents the merged state of both original
    accumulators.
*/
template<
    typename Argument,
    typename Result>
inline Max<Argument, Result>& Max<Argument, Result>::operator|=(
    Max const& other)
{
    operator()(other._max);
    return *this;
}


/*!
    @ingroup    fern_algorithm_accumulator_group
    @brief      Merge @a lhs with @a rhs.

    The resulting accumulator represents the merged state of both original
    accumulators.
*/
template<
    typename Argument,
    typename Result>
inline Max<Argument, Result> operator|(
    Max<Argument, Result> const& lhs,
    Max<Argument, Result> const& rhs)
{
    return Max<Argument, Result>(lhs) |= rhs;
}

} // namespace accumulator
} // namespace algorithm
} // namespace fern
