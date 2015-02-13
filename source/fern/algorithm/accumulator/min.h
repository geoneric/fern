#pragma once
#include <algorithm>
#include <limits>


namespace fern {
namespace algorithm {
namespace accumulator {

/*!
    @ingroup    fern_algorithm_accumulator_group
    @brief      Accumulator that calculates the minimum of the added values.
*/
template<
    typename Argument,
    typename Result=Argument>
class Min
{

public:

    /*!
        @brief      During the addition of values, this accumulator's
                    layered minimum value will never go out of range. It
                    is of type @a Argument.
    */
    static bool const out_of_range_risk{false};

                   Min                 ();

    explicit       Min                 (Argument const& value);

    void           operator=           (Argument const& value);

    void           operator()          (Argument const& value);

    Result         operator()          () const;

    Min&           operator|=          (Min const& other);

private:

    Argument       _min;

};


/*!
    @brief      Default construct an instance.

    The layered minimum value is initialized with
    std::numeric_limits<Argument>::max().
*/
template<
    typename Argument,
    typename Result>
inline Min<Argument, Result>::Min()

    : _min(std::numeric_limits<Argument>::max())

{
}


/*!
    @brief      Construct an instance.

    The layered minimum value is initialized with @a value.
*/
template<
    typename Argument,
    typename Result>
inline Min<Argument, Result>::Min(
    Argument const& value)

    : _min(value)

{
}


/*!
    @brief      Assign @a value to the instance.

    This initializes the instance to be reused again.
*/
template<
    typename Argument,
    typename Result>
inline void Min<Argument, Result>::operator=(
    Argument const& value)
{
    _min = value;
}


/*!
    @brief      Add @a value to the instance.

    In case @a value is smaller than the layered minimum value, it
    is assigned to the layered minimum value. Otherwise, the layered
    minimum value is unchanged.
*/
template<
    typename Argument,
    typename Result>
inline void Min<Argument, Result>::operator()(
    Argument const& value)
{
    _min = std::min(_min, value);
}


/*!
    @brief      Return the minimum value seen until now.

    The layered minimum value is cast to the @a Result type.
*/
template<
    typename Argument,
    typename Result>
inline Result Min<Argument, Result>::operator()() const
{
    return static_cast<Result>(_min);
}


/*!
    @brief      Merge @a other with *this.

    The resulting accumulator represents the merged state of both original
    accumulators.
*/
template<
    typename Argument,
    typename Result>
inline Min<Argument, Result>& Min<Argument, Result>::operator|=(
    Min const& other)
{
    operator()(other._min);
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
inline Min<Argument, Result> operator|(
    Min<Argument, Result> const& lhs,
    Min<Argument, Result> const& rhs)
{
    return Min<Argument, Result>(lhs) |= rhs;
}

} // namespace accumulator
} // namespace algorithm
} // namespace fern
