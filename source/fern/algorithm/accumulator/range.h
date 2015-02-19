#pragma once
#include "fern/algorithm/accumulator/min.h"
#include "fern/algorithm/accumulator/max.h"


namespace fern {
namespace algorithm {
namespace accumulator {

/*!
    @ingroup    fern_algorithm_accumulator_group
    @brief      Accumulator that calculates the range of the added values.
*/
template<
    typename Argument,
    typename Result=Argument>
class Range
{

public:

    /*!
        @brief      During the addition of values, this accumulator's
                    layered extreme values will never go out of range. They
                    are of type @a Argument.
    */
    static bool const out_of_range_risk{false};

                   Range               ();

    explicit       Range               (Argument const& value);

    void           operator=           (Argument const& value);

    void           operator()          (Argument const& value);

    Result         operator()          () const;

    Range&         operator|=          (Range const& other);

private:

    Min<Argument, Result> _min;

    Max<Argument, Result> _max;

};


/*!
    @brief      Default construct an instance.
*/
template<
    typename Argument,
    typename Result>
inline Range<Argument, Result>::Range()

    : _min(),
      _max()

{
}


/*!
    @brief      Construct an instance initializing it with @a value.
*/
template<
    typename Argument,
    typename Result>
inline Range<Argument, Result>::Range(
    Argument const& value)

    : _min(value),
      _max(value)

{
}


/*!
    @brief      Assign @a value to the instance.

    This initializes the instance to be reused again.
*/
template<
    typename Argument,
    typename Result>
inline void Range<Argument, Result>::operator=(
    Argument const& value)
{
    _min = value;
    _max = value;
}


/*!
    @brief      Add @a value to the instance.

    In case @a value is smaller than the layered minimum value, it is assigned
    to the layered minimum value. Otherwise, the layered minimum value is
    unchanged.

    In case @a value is larger than the layered maximum value, it is assigned
    to the layered maximum value. Otherwise, the layered maximum value is
    unchanged.
*/
template<
    typename Argument,
    typename Result>
inline void Range<Argument, Result>::operator()(
    Argument const& value)
{
    _min(value);
    _max(value);
}


/*!
    @brief      Return the range in the values seen until now.

    The range is cast to the @a Result type.
*/
template<
    typename Argument,
    typename Result>
inline Result Range<Argument, Result>::operator()() const
{
    return _max() - _min();
}


/*!
    @brief      Merge @a other with *this.

    The resulting accumulator represents the merged state of both original
    accumulators.
*/
template<
    typename Argument,
    typename Result>
inline Range<Argument, Result>& Range<Argument, Result>::operator|=(
    Range const& other)
{
    _min |= other._min;
    _max |= other._max;
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
inline Range<Argument, Result> operator|(
    Range<Argument, Result> const& lhs,
    Range<Argument, Result> const& rhs)
{
    return Range<Argument, Result>(lhs) |= rhs;
}

} // namespace accumulator
} // namespace algorithm
} // namespace fern
