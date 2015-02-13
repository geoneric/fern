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

    Result         operator()          () const;

    Count&         operator|=          (Count const& other);

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
