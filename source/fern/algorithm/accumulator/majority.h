#pragma once
#include "fern/algorithm/accumulator/histogram.h"


namespace fern {
namespace algorithm {
namespace accumulator {

/*!
    @ingroup    fern_algorithm_accumulator_group
    @brief      Accumulator that calculates the most occurring value.
*/
template<
    typename Argument,
    typename Result=Argument>
class Majority
{

public:

    /*!
        @brief      During the addition of values, this accumulator's
                    layered values will never go out of range. They are
                    of type @a Argument.
    */
    static bool const out_of_range_risk{false};

                   Majority            ();

    explicit       Majority            (Argument const& value);

    void           operator=           (Argument const& value);

    void           operator()          (Argument const& value);

    Result         operator()          () const;

    Majority&      operator|=          (Majority const& other);

private:

    Histogram<Argument> _histogram;

};


/*!
    @brief      Default construct an instance.
*/
template<
    typename Argument,
    typename Result>
inline Majority<Argument, Result>::Majority()

    : _histogram(0)

{
}


/*!
    @brief      Construct an instance and add @a value.
*/
template<
    typename Argument,
    typename Result>
inline Majority<Argument, Result>::Majority(
    Argument const& value)

    : _histogram{value}

{
}


/*!
    @brief      Assign @a value to the instance.

    This initializes the instance to be reused again.
*/
template<
    typename Argument,
    typename Result>
inline void Majority<Argument, Result>::operator=(
    Argument const& value)
{
    _histogram = {value};
}


/*!
    @brief      Add @a value to the instance.

    The layered count for @a value is increased by one.
*/
template<
    typename Argument,
    typename Result>
inline void Majority<Argument, Result>::operator()(
    Argument const& value)
{
    _histogram(value);
}


/*!
    @brief      Return the most occurring value of the values added until now.

    TODO Which value is returned if more than one value occurs most.
*/
template<
    typename Argument,
    typename Result>
inline Result Majority<Argument, Result>::operator()() const
{
    assert(!_histogram.empty());
    return _histogram.majority();
}


/*!
    @brief      Merge @a other with *this.

    The resulting accumulator represents the merged state of both original
    accumulators.
*/
template<
    typename Argument,
    typename Result>
inline Majority<Argument, Result>& Majority<Argument, Result>::operator|=(
    Majority const& other)
{
    _histogram |= other._histogram;
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
inline Majority<Argument, Result> operator|(
    Majority<Argument, Result> const& lhs,
    Majority<Argument, Result> const& rhs)
{
    return Majority<Argument, Result>(lhs) |= rhs;
}

} // namespace accumulator
} // namespace algorithm
} // namespace fern
