#pragma once
#include <cassert>
#include "fern/algorithm/accumulator/sum.h"


namespace fern {
namespace algorithm {
namespace accumulator {

/*!
    @ingroup    fern_algorithm_accumulator_group
    @brief      Accumulator that calculates the mean of the added values.
*/
template<
    typename Argument,
    typename Result=Argument>
class Mean
{

public:

    /*!
        @brief      During the addition of values, this accumulator's
                    layered Sum accumulator may go out of range. It is
                    of type @a Result.
    */
    static bool const out_of_range_risk{true};

                   Mean                ();

    explicit       Mean                (Argument const& value);

    void           operator=           (Argument const& value);

    void           operator()          (Argument const& value);

    Result         operator()          () const;

private:

    Sum<Argument, Result> _sum;

    size_t         _count;

};


/*!
    @brief      Default construct an instance.
    @warning    Don't ask for the result of a default constructed instance.
                Since the layered count is initialized with 0, dividing the
                sum of values by the count will 'fail'.

    The layered Sum accumulator is default constructed, and the layered
    count is initialized with 0.
*/
template<
    typename Argument,
    typename Result>
inline Mean<Argument, Result>::Mean()

    : _sum(),
      _count(0u)

{
}


/*!
    @brief      Construct an instance.

    The layered Sum accumulator is constructed passing @a value, and the
    layered count is initialized with 1.
*/
template<
    typename Argument,
    typename Result>
inline Mean<Argument, Result>::Mean(
    Argument const& value)

    : _sum(value),
      _count(1u)

{
}


/*!
    @brief      Assign @a value to the instance.

    This initializes the instance to be reused again.
*/
template<
    typename Argument,
    typename Result>
inline void Mean<Argument, Result>::operator=(
    Argument const& value)
{
    _sum = value;
    _count = 1u;
}


/*!
    @brief      Add @a value to the instance.

    A running sum of all values added is stored in the layered Sum
    accumulator. The layered count is increased by one.
*/
template<
    typename Argument,
    typename Result>
inline void Mean<Argument, Result>::operator()(
    Argument const& value)
{
    _sum(value);
    ++_count;
}


/*!
    @brief      Return the mean value given all values added until now.

    The result is cast to the @a Result type.
*/
template<
    typename Argument,
    typename Result>
inline Result Mean<Argument, Result>::operator()() const
{
    assert(_count > 0);

    // _sum() returns result in Result. Dividing by _count may implicitly cast
    // it to some other type.
    return Result(_sum() / _count);
}

} // namespace accumulator
} // namespace algorithm
} // namespace fern
