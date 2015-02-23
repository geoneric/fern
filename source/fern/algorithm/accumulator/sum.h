#pragma once


namespace fern {
namespace algorithm {
namespace accumulator {

/*!
    @ingroup    fern_algorithm_accumulator_group
    @brief      Accumulator that calculates the sum of the added values.
*/
template<
    typename Argument,
    typename Result=Argument>
class Sum
{

public:

    /*!
        @brief      During the addition of values, this accumulator's
                    layered sum may go out of range. It is of type @a Result.
    */
    static bool const out_of_range_risk{true};

                   Sum                 ();

    explicit       Sum                 (Argument const& value);

                   Sum                 (Sum const& other)=default;

                   Sum                 (Sum&& other)=default;

                   ~Sum                ()=default;

    Sum&           operator=           (Sum const& other)=default;

    Sum&           operator=           (Sum&& other)=default;

    void           operator=           (Argument const& value);

    void           operator()          (Argument const& value);

    Result         operator()          () const;

    template<
        template<typename, typename, typename> class OutOfRangePolicy>
    bool           operator()          (Argument const& value);

    Sum&           operator|=          (Sum const& other);

    template<
        template<typename, typename, typename> class OutOfRangePolicy>
    bool           operator|=          (Sum const& other);

private:

    Result         _sum;

};


/*!
    @brief      Default construct an instance.

    The layered sum is initialized with 0.
*/
template<
    typename Argument,
    typename Result>
inline Sum<Argument, Result>::Sum()

    : _sum(0)

{
}


/*!
    @brief      Construct an instance.

    The layered sum is initialized with @a value.
*/
template<
    typename Argument,
    typename Result>
inline Sum<Argument, Result>::Sum(
    Argument const& value)

    : _sum(value)

{
}


/*!
    @brief      Assign @a value to the instance.

    This initializes the instance to be reused again.
*/
template<
    typename Argument,
    typename Result>
inline void Sum<Argument, Result>::operator=(
    Argument const& value)
{
    _sum = value;
}


/*!
    @brief      Add @a value to the instance.

    @a value is added to the layered sum.
*/
template<
    typename Argument,
    typename Result>
inline void Sum<Argument, Result>::operator()(
    Argument const& value)
{
    _sum += value;
}


/*!
    @brief      Return the sum of values seen until now.
*/
template<
    typename Argument,
    typename Result>
inline Result Sum<Argument, Result>::operator()() const
{
    return _sum;
}


/*!
    @brief      Add @a value to the instance.
    @tparam     OutOfRangePolicy Policy to use during check for out of range
                result.
    @return     Whether or not the layered sum went out of range according
                to @a OutOfRangePolicy.

    @a value is added to the layered sum.
*/
template<
    typename Argument,
    typename Result>
template<
    template<typename, typename, typename> class OutOfRangePolicy>
inline bool Sum<Argument, Result>::operator()(
    Argument const& value)
{
    Result previous_sum{_sum};
    _sum += value;

    return OutOfRangePolicy<Result, Argument, Result>::within_range(
        previous_sum, value, _sum);
}


/*!
    @brief      Merge @a other with *this.

    The resulting accumulator represents the merged state of both original
    accumulators.
*/
template<
    typename Argument,
    typename Result>
inline Sum<Argument, Result>& Sum<Argument, Result>::operator|=(
    Sum const& other)
{
    _sum += other._sum;
    return *this;
}


template<
    typename Argument,
    typename Result>
template<
    template<typename, typename, typename> class OutOfRangePolicy>
inline bool Sum<Argument, Result>::operator|=(
    Sum const& other)
{
    return operator()<OutOfRangePolicy>(other._sum);
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
inline Sum<Argument, Result> operator|(
    Sum<Argument, Result> const& lhs,
    Sum<Argument, Result> const& rhs)
{
    return Sum<Argument, Result>(lhs) |= rhs;
}

} // namespace accumulator
} // namespace algorithm
} // namespace fern
