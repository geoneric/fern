// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <vector>
#include "fern/algorithm/accumulator/mean.h"


namespace fern {
namespace algorithm {
namespace accumulator {

/*!
    @ingroup    fern_algorithm_accumulator_group
    @brief      Accumulator that calculates the variance of added values.

    The variance is a summary statistic. It is a measure of spread of a
    distribution of values.
*/
template<
    typename Argument,
    typename Result=Argument>
class Variance
{

public:

    /*!
        @brief      During the addition of values, this accumulator's
                    layered Mean accumulator may go out of range.
    */
    static bool const out_of_range_risk{true};

                   Variance            ();

    explicit       Variance            (Argument const& value);

    void           operator=           (Argument const& value);

    void           operator()          (Argument const& value);

    Result         operator()          () const;

    Variance&      operator|=          (Variance const& other);

private:

    // TODO Once we have a full blown square algorithm we must use that
    //      instead of this one.
    template<
        typename T>
    inline constexpr T square(
        T const& value)
    {
        return value * value;
    }

    Mean<Argument, Result> _mean;

    std::vector<Argument> _values;

};


/*!
    @brief      Default construct an instance.
*/
template<
    typename Argument,
    typename Result>
inline Variance<Argument, Result>::Variance()

    : _mean(),
      _values()

{
}


/*!
    @brief      Construct an instance.

    The layered Mean accumulator and collection of values are constructed
    passing @a value.
*/
template<
    typename Argument,
    typename Result>
inline Variance<Argument, Result>::Variance(
    Argument const& value)

    : _mean(value),
      _values{value}

{
}


/*!
    @brief      Assign @a value to the instance.

    This initializes the instance to be reused again.
*/
template<
    typename Argument,
    typename Result>
inline void Variance<Argument, Result>::operator=(
    Argument const& value)
{
    _mean = value;
    _values = {value};
}


/*!
    @brief      Add @a value to the instance.
*/
template<
    typename Argument,
    typename Result>
inline void Variance<Argument, Result>::operator()(
    Argument const& value)
{
    _mean(value);
    _values.emplace_back(value);
}


/*!
    @brief      Return the median value of the values added until now.
*/
template<
    typename Argument,
    typename Result>
inline Result Variance<Argument, Result>::operator()() const
{
    assert(!_values.empty());

    Result mean{_mean()};
    Result result{0};

    for(auto const& value: _values) {
        result += square(mean - static_cast<Result>(value));
    }

    return result / static_cast<Result>(_values.size());
}


/*!
    @brief      Merge @a other with *this.

    The resulting accumulator represents the merged state of both original
    accumulators.
*/
template<
    typename Argument,
    typename Result>
inline Variance<Argument, Result>& Variance<Argument, Result>::operator|=(
    Variance const& other)
{
    _mean |= other._mean;

    _values.reserve(_values.size() + other._values.size());
    _values.insert(_values.end(), other._values.begin(), other._values.end());

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
inline Variance<Argument, Result> operator|(
    Variance<Argument, Result> const& lhs,
    Variance<Argument, Result> const& rhs)
{
    return Variance<Argument, Result>(lhs) |= rhs;
}

} // namespace accumulator
} // namespace algorithm
} // namespace fern
