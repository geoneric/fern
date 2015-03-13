#pragma once
#include <cassert>
#include <cstddef>
#include "fern/algorithm/accumulator/count.h"
#include "fern/algorithm/accumulator/sum.h"


namespace fern {
namespace algorithm {
namespace accumulator {

/*!
    @ingroup    fern_algorithm_accumulator_group
    @brief      Accumulator that calculates the mean of the added values.

    The mean is a summary statistic. It is a measure of location of the
    center of a distribution of values.
*/
template<
    typename Argument,
    typename Result=Argument>
class Mean
{

public:

    /*!
        @brief      During the addition of values, this accumulator's
                    layered Sum accumulator may go out of range.

        In theory, the layered Count accumulator may also go out of range,
        but as long as values from a single collection are added, this will
        never happen. It contains a state variable of type size_t.
        Tests involving an out-of-range policy do not test the count.
    */
    static bool const out_of_range_risk{
        Sum<Argument, Result>::out_of_range_risk};

                   Mean                ();

    explicit       Mean                (Argument const& value);

    void           operator=           (Argument const& value);

    void           operator()          (Argument const& value);

    template<
        template<typename, typename, typename> class OutOfRangePolicy>
    bool           operator()          (Argument const& value);

    Result         operator()          () const;

    Mean&          operator|=          (Mean const& other);

    template<
        template<typename, typename, typename> class OutOfRangePolicy>
    bool           operator|=          (Mean const& other);

private:

    Sum<Argument, Result> _sum;

    Count<Argument, size_t> _count;

};


/*!
    @brief      Default construct an instance.
    @warning    Don't ask for the result of a default constructed instance.
                Since the layered count is initialized with 0, dividing the
                sum of values by the count will 'fail'.

    The layered Sum and Count accumulators are default constructed.
*/
template<
    typename Argument,
    typename Result>
inline Mean<Argument, Result>::Mean()

    : _sum(),
      _count()

{
}


/*!
    @brief      Construct an instance.

    The layered Sum and Count accumulators are constructed passing @a value.
*/
template<
    typename Argument,
    typename Result>
inline Mean<Argument, Result>::Mean(
    Argument const& value)

    : _sum(value),
      _count(value)

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
    _count = value;
}


/*!
    @brief      Add @a value to the instance.

    A running sum of all values added is stored in the layered Sum
    accumulator. The number of values added is stored in the layered Count
    accumulator.
*/
template<
    typename Argument,
    typename Result>
inline void Mean<Argument, Result>::operator()(
    Argument const& value)
{
    _sum(value);
    _count(value);
}


template<
    typename Argument,
    typename Result>
template<
    template<typename, typename, typename> class OutOfRangePolicy>
inline bool Mean<Argument, Result>::operator()(
    Argument const& value)
{
    _count(value);
    return _sum.operator()<OutOfRangePolicy>(value);
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
    assert(_count() > 0);

    // _sum() returns result in Result. Dividing by _count may implicitly cast
    // it to some other type.
    return Result(_sum() / _count());
}


/*!
    @brief      Merge @a other with *this.

    The resulting accumulator represents the merged state of both original
    accumulators.
*/
template<
    typename Argument,
    typename Result>
inline Mean<Argument, Result>& Mean<Argument, Result>::operator|=(
    Mean const& other)
{
    _sum |= other._sum;
    _count |= other._count;
    return *this;
}


template<
    typename Argument,
    typename Result>
template<
    template<typename, typename, typename> class OutOfRangePolicy>
inline bool Mean<Argument, Result>::operator|=(
    Mean const& other)
{
    _count |= other._count;
    return _sum.operator|=<OutOfRangePolicy>(other._sum);
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
inline Mean<Argument, Result> operator|(
    Mean<Argument, Result> const& lhs,
    Mean<Argument, Result> const& rhs)
{
    return Mean<Argument, Result>(lhs) |= rhs;
}

} // namespace accumulator
} // namespace algorithm
} // namespace fern
