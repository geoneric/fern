#pragma once
#include <map>


namespace fern {
namespace algorithm {
namespace accumulator {

/*!
    @ingroup    fern_algorithm_accumulator_group
    @brief      Class for storing count per value.
*/
template<
    typename T>
class Histogram
{

public:

                   Histogram           ();

    explicit       Histogram           (T const& value);

    void           operator=           (T const& value);

    void           operator()          (T const& value);

    Histogram&     operator|=          (Histogram const& other);

    bool           empty               () const;

    size_t         size                () const;

    T const&       minority            () const;

    T const&       majority            () const;

private:

    std::map<T, size_t> _count;

};


/*!
    @brief      Default construct an instance.
*/
template<
    typename T>
inline Histogram<T>::Histogram()

    : _count()

{
}


/*!
    @brief      Construct an instance and add @a value.
*/
template<
    typename T>
inline Histogram<T>::Histogram(
    T const& value)

    : _count{{value, 1}}

{
}


/*!
    @brief      Assign @a value to the instance.

    This initializes the instance to be reused again.
*/
template<
    typename T>
inline void Histogram<T>::operator=(
    T const& value)
{
    _count = {{value, 1}};
}


/*!
    @brief      Add @a value to the instance.

    The layered count for @a value is increased by one.
*/
template<
    typename T>
inline void Histogram<T>::operator()(
    T const& value)
{
    ++_count[value];
}


/*!
    @brief      Merge @a other with *this.
*/
template<
    typename T>
inline Histogram<T>& Histogram<T>::operator|=(
    Histogram const& other)
{
    // TODO _count |= other._count;
    return *this;
}


template<
    typename T>
inline bool Histogram<T>::empty() const
{
    return _count.empty();
}


template<
    typename T>
inline size_t Histogram<T>::size() const
{
    return _count.size();
}


template<
    typename T>
inline T const& Histogram<T>::minority() const
{
    assert(!empty());

    // Find value with the lowest count.

    return _count.begin()->first;
}


template<
    typename T>
inline T const& Histogram<T>::majority() const
{
    assert(!empty());

    // Find value with the highest count.

    return _count.begin()->first;
}


/*!
    @ingroup    fern_algorithm_accumulator_group
    @brief      Merge @a lhs with @a rhs.
*/
template<
    typename T>
inline Histogram<T> operator|(
    Histogram<T> const& lhs,
    Histogram<T> const& rhs)
{
    return Histogram<T>(lhs) |= rhs;
}

} // namespace accumulator
} // namespace algorithm
} // namespace fern
