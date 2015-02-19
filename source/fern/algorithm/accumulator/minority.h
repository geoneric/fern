#pragma once
#include "fern/algorithm/accumulator/histogram.h"


namespace fern {
namespace algorithm {
namespace accumulator {

/*!
    @ingroup    fern_algorithm_accumulator_group
    @brief      Accumulator that calculates the least occurring value.
*/
template<
    typename Argument,
    typename Result=Argument>
class Minority
{

public:

    /*!
        @brief      During the addition of values, this accumulator's
                    layered values will never go out of range. They are
                    of type @a Argument.
    */
    static bool const out_of_range_risk{false};

                   Minority            ();

    explicit       Minority            (Argument const& value);

    void           operator=           (Argument const& value);

    void           operator()          (Argument const& value);

    Result         operator()          () const;

    Minority&      operator|=          (Minority const& other);

private:

    Histogram<Argument> _histogram;

};


/*!
    @brief      Default construct an instance.
*/
template<
    typename Argument,
    typename Result>
inline Minority<Argument, Result>::Minority()

    : _histogram(0)

{
}


/*!
    @brief      Construct an instance and add @a value.
*/
template<
    typename Argument,
    typename Result>
inline Minority<Argument, Result>::Minority(
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
inline void Minority<Argument, Result>::operator=(
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
inline void Minority<Argument, Result>::operator()(
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
inline Result Minority<Argument, Result>::operator()() const
{
    assert(!_histogram.empty());
    return _histogram.minority();
}


/*!
    @brief      Merge @a other with *this.

    The resulting accumulator represents the merged state of both original
    accumulators.
*/
template<
    typename Argument,
    typename Result>
inline Minority<Argument, Result>& Minority<Argument, Result>::operator|=(
    Minority const& other)
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
inline Minority<Argument, Result> operator|(
    Minority<Argument, Result> const& lhs,
    Minority<Argument, Result> const& rhs)
{
    return Minority<Argument, Result>(lhs) |= rhs;
}

} // namespace accumulator
} // namespace algorithm
} // namespace fern
