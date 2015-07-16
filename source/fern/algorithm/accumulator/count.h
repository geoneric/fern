// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once


namespace fern {
namespace algorithm {
namespace accumulator {

/*!
    @ingroup    fern_algorithm_accumulator_group
    @brief      Accumulator that calculates the number of added values.
*/
template<
    typename Argument,
    typename Result=Argument>
class Count
{

public:

    /*!
        @brief      During the addition of values, this accumulator's
                    layered count may go out of range. It is of type
                    @a Result.
    */
    static bool const out_of_range_risk{true};

                   Count               ();

    explicit       Count               (Argument const& value);

    void           operator=           (Argument const& value);

    void           operator()          (Argument const& value);

    template<
        template<typename, typename, typename> class OutOfRangePolicy>
    bool           operator()          (Argument const& value);

    Result         operator()          () const;

    Count&         operator|=          (Count const& other);

    template<
        template<typename, typename, typename> class OutOfRangePolicy>
    bool           operator|=          (Count const& other);

private:

    Result         _count;

};


/*!
    @brief      Default construct an instance.

    The layered count is initialized with 0.
*/
template<
    typename Argument,
    typename Result>
inline Count<Argument, Result>::Count()

    : _count(0)

{
}


/*!
    @brief      Construct an instance.

    The layered count is initialized with 1.
*/
template<
    typename Argument,
    typename Result>
inline Count<Argument, Result>::Count(
    Argument const& /* value */)

    : _count(1)

{
}


/*!
    @brief      Assign @a value to the instance.

    This initializes the instance to be reused again.
*/
template<
    typename Argument,
    typename Result>
inline void Count<Argument, Result>::operator=(
    Argument const& /* value */)
{
    _count = 1;
}


/*!
    @brief      Add @a value to the instance.

    The layered count is increased by one.
*/
template<
    typename Argument,
    typename Result>
inline void Count<Argument, Result>::operator()(
    Argument const& /* value */)
{
    ++_count;
}


template<
    typename Argument,
    typename Result>
template<
    template<typename, typename, typename> class OutOfRangePolicy>
inline bool Count<Argument, Result>::operator()(
    Argument const& /* value */)
{
    Result previous_count{_count};
    ++_count;

    return OutOfRangePolicy<Result, Argument, Result>::within_range(
        previous_count, 1, _count);
}


/*!
    @brief      Return the number of values seen until now.
*/
template<
    typename Argument,
    typename Result>
inline Result Count<Argument, Result>::operator()() const
{
    return _count;
}


/*!
    @brief      Merge @a other with *this.

    The resulting accumulator represents the merged state of both original
    accumulators.
*/
template<
    typename Argument,
    typename Result>
inline Count<Argument, Result>& Count<Argument, Result>::operator|=(
    Count const& other)
{
    _count += other._count;
    return *this;
}


template<
    typename Argument,
    typename Result>
template<
    template<typename, typename, typename> class OutOfRangePolicy>
inline bool Count<Argument, Result>::operator|=(
    Count const& /* other */)
{
    // Value doesn't matter. We're counting, not summing.
    return operator()<OutOfRangePolicy>(Argument{});
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
inline Count<Argument, Result> operator|(
    Count<Argument, Result> const& lhs,
    Count<Argument, Result> const& rhs)
{
    return Count<Argument, Result>(lhs) |= rhs;
}

} // namespace accumulator
} // namespace algorithm
} // namespace fern
