#pragma once
#include <algorithm>
#include <unordered_map>


namespace fern {
namespace algorithm {
namespace accumulator {
namespace detail {

/*!
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

    T const&       mode                () const;

private:

    std::unordered_map<T, size_t> _count;

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
    for(auto const& pair: other._count) {
        auto position = _count.find(pair.first);
        if(position == _count.end()) {
            _count.insert(pair);
        }
        else {
            position->second += pair.second;
        }
    }

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


/*!
    @brief      Return the mode of the histogram.

    In case there are multiple modal values, it is undefined which one is
    returned.
*/
template<
    typename T>
inline T const& Histogram<T>::mode() const
{
    assert(!empty());

    // For speed, _count is an unordered_map instead of a regular map.
    // Because of that, we can't make promisses about which value is returned
    // in the multi-modal case. We don't know in which order the values in
    // the map are visited by the algorithm below.

    // Find a value with the highest count.
    auto result = std::max_element(_count.begin(), _count.end(),
        [](auto const& lhs, auto const& rhs) {
            return lhs.second < rhs.second; });

    return result->first;
}


/*!
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

} // namespace detail
} // namespace accumulator
} // namespace algorithm
} // namespace fern
