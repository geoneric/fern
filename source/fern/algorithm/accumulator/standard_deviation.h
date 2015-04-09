// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <cmath>
#include "fern/algorithm/accumulator/variance.h"


namespace fern {
namespace algorithm {
namespace accumulator {

/*!
    @ingroup    fern_algorithm_accumulator_group
    @brief      Accumulator that calculates the standard deviation of
                added values.

    The standard deviation is a summary statistic. It is a measure of
    spread of a distribution of values.
*/
template<
    typename Argument,
    typename Result=Argument>
class StandardDeviation
{

public:

    /*!
        @brief      During the addition of values, this accumulator's
                    layered Variance accumulator may go out of range.
    */
    static bool const out_of_range_risk{true};

                   StandardDeviation   ();

    explicit       StandardDeviation   (Argument const& value);

    void           operator=           (Argument const& value);

    void           operator()          (Argument const& value);

    Result         operator()          () const;

    StandardDeviation&
                   operator|=          (StandardDeviation const& other);

private:

    Variance<Argument, Result> _variance;

};


/*!
    @brief      Default construct an instance.
*/
template<
    typename Argument,
    typename Result>
inline StandardDeviation<Argument, Result>::StandardDeviation()

    : _variance()

{
}


/*!
    @brief      Construct an instance.

    The layered Variance accumulator is initialized with @a value.
*/
template<
    typename Argument,
    typename Result>
inline StandardDeviation<Argument, Result>::StandardDeviation(
    Argument const& value)

    : _variance{value}

{
}


/*!
    @brief      Assign @a value to the instance.

    This initializes the instance to be reused again.
*/
template<
    typename Argument,
    typename Result>
inline void StandardDeviation<Argument, Result>::operator=(
    Argument const& value)
{
    _variance = {value};
}


/*!
    @brief      Add @a value to the instance.
*/
template<
    typename Argument,
    typename Result>
inline void StandardDeviation<Argument, Result>::operator()(
    Argument const& value)
{
    _variance(value);
}


/*!
    @brief      Return the standard deviation of the values added until now.
*/
template<
    typename Argument,
    typename Result>
inline Result StandardDeviation<Argument, Result>::operator()() const
{
    return static_cast<Result>(std::sqrt(_variance()));
}


/*!
    @brief      Merge @a other with *this.

    The resulting accumulator represents the merged state of both original
    accumulators.
*/
template<
    typename Argument,
    typename Result>
inline StandardDeviation<Argument, Result>&
    StandardDeviation<Argument, Result>::operator|=(
        StandardDeviation const& other)
{
    _variance |= other._variance;
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
inline StandardDeviation<Argument, Result> operator|(
    StandardDeviation<Argument, Result> const& lhs,
    StandardDeviation<Argument, Result> const& rhs)
{
    return StandardDeviation<Argument, Result>(lhs) |= rhs;
}

} // namespace accumulator
} // namespace algorithm
} // namespace fern
