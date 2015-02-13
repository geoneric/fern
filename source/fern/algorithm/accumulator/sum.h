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

    void           operator=           (Argument const& value);

    void           operator()          (Argument const& value);

    Result         operator()          () const;

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

} // namespace accumulator
} // namespace algorithm
} // namespace fern
