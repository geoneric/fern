// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <algorithm>
#include <vector>


namespace fern {
namespace algorithm {
namespace accumulator {

/*!
    @ingroup    fern_algorithm_accumulator_group
    @brief      Accumulator that calculates the median of added values.

    The median is a summary statistic. It is a measure of location of the
    center of a distribution of values.
*/
template<
    typename Argument,
    typename Result=Argument>
class Median
{

public:

    /*!
        @brief      During the addition of values, this accumulator's
                    layered values will never go out of range. They are
                    of type @a Argument.
    */
    static bool const out_of_range_risk{false};

                   Median              ();

    explicit       Median              (Argument const& value);

    void           operator=           (Argument const& value);

    void           operator()          (Argument const& value);

    Result         operator()          () const;

    Median&        operator|=          (Median const& other);

private:

    mutable std::vector<Argument> _values;

};


/*!
    @brief      Default construct an instance.

    The layered collection of values is default constructed.
*/
template<
    typename Argument,
    typename Result>
inline Median<Argument, Result>::Median()

    : _values(0)

{
}


/*!
    @brief      Construct an instance.

    The layered collection of values is initialized with @a value.
*/
template<
    typename Argument,
    typename Result>
inline Median<Argument, Result>::Median(
    Argument const& value)

    : _values{value}

{
}


/*!
    @brief      Assign @a value to the instance.

    This initializes the instance to be reused again.
*/
template<
    typename Argument,
    typename Result>
inline void Median<Argument, Result>::operator=(
    Argument const& value)
{
    _values = {value};
}


/*!
    @brief      Add @a value to the instance.

    The layered count is increased by one.
    @a value is added to the layered collection of values.
*/
template<
    typename Argument,
    typename Result>
inline void Median<Argument, Result>::operator()(
    Argument const& value)
{
    _values.emplace_back(value);
}


/*!
    @brief      Return the median value of the values added until now.

    In case the number of values in the layered collection is odd, the
    middle value is returned.

    In case the number of values in the layered collection is even, the
    mean of the two middle values is returned.
*/
template<
    typename Argument,
    typename Result>
inline Result Median<Argument, Result>::operator()() const
{
    assert(!_values.empty());
    std::sort(_values.begin(), _values.end());
    size_t center_index{_values.size() / 2};

    Result result;

    if(_values.size() % 2 == 1) {
        // Odd. Return center value.
        result = static_cast<Result>(_values[center_index]);
    }
    else {
        // Even. Return mean of two center values.
        result = static_cast<Result>(_values[center_index - 1] +
            _values[center_index]) / 2;
    }

    return result;
}


/*!
    @brief      Merge @a other with *this.

    The resulting accumulator represents the merged state of both original
    accumulators.
*/
template<
    typename Argument,
    typename Result>
inline Median<Argument, Result>& Median<Argument, Result>::operator|=(
    Median const& other)
{
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
inline Median<Argument, Result> operator|(
    Median<Argument, Result> const& lhs,
    Median<Argument, Result> const& rhs)
{
    return Median<Argument, Result>(lhs) |= rhs;
}

} // namespace accumulator
} // namespace algorithm
} // namespace fern
