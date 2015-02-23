#pragma once
#include "fern/algorithm/accumulator/detail/frequency_table.h"


namespace fern {
namespace algorithm {
namespace accumulator {

/*!
    @ingroup    fern_algorithm_accumulator_group
    @brief      Accumulator that calculates the mode of the added values.

    The mode of a collection of values is the value that occurs most.

    The mode is a summary statistic. It is a measure of location of a
    distribution of values.
*/
template<
    typename Argument,
    typename Result=Argument>
class Mode
{

public:

    /*!
        @brief      During the addition of values, this accumulator's
                    layered values will never go out of range. They are
                    of type @a Argument.
    */
    static bool const out_of_range_risk{false};

                   Mode                ();

    explicit       Mode                (Argument const& value);

    void           operator=           (Argument const& value);

    void           operator()          (Argument const& value);

    Result         operator()          () const;

    Mode&          operator|=          (Mode const& other);

private:

    detail::FrequencyTable<Argument> _frequency_table;

};


/*!
    @brief      Default construct an instance.
*/
template<
    typename Argument,
    typename Result>
inline Mode<Argument, Result>::Mode()

    : _frequency_table(0)

{
}


/*!
    @brief      Construct an instance and add @a value.
*/
template<
    typename Argument,
    typename Result>
inline Mode<Argument, Result>::Mode(
    Argument const& value)

    : _frequency_table{value}

{
}


/*!
    @brief      Assign @a value to the instance.

    This initializes the instance to be reused again.
*/
template<
    typename Argument,
    typename Result>
inline void Mode<Argument, Result>::operator=(
    Argument const& value)
{
    _frequency_table = {value};
}


/*!
    @brief      Add @a value to the instance.

    The layered count for @a value is increased by one.
*/
template<
    typename Argument,
    typename Result>
inline void Mode<Argument, Result>::operator()(
    Argument const& value)
{
    _frequency_table(value);
}


/*!
    @brief      Return the most occurring value of the values added until now.

    TODO Which value is returned in multi-modal case?
*/
template<
    typename Argument,
    typename Result>
inline Result Mode<Argument, Result>::operator()() const
{
    assert(!_frequency_table.empty());
    return _frequency_table.mode();
}


/*!
    @brief      Merge @a other with *this.

    The resulting accumulator represents the merged state of both original
    accumulators.
*/
template<
    typename Argument,
    typename Result>
inline Mode<Argument, Result>& Mode<Argument, Result>::operator|=(
    Mode const& other)
{
    _frequency_table |= other._frequency_table;
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
inline Mode<Argument, Result> operator|(
    Mode<Argument, Result> const& lhs,
    Mode<Argument, Result> const& rhs)
{
    return Mode<Argument, Result>(lhs) |= rhs;
}

} // namespace accumulator
} // namespace algorithm
} // namespace fern
