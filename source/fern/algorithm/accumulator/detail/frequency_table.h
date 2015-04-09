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
#include <unordered_map>


namespace fern {
namespace algorithm {
namespace accumulator {
namespace detail {

/*!
    @brief      Class for storing frequency per value.
*/
template<
    typename T>
class FrequencyTable
{

public:

                   FrequencyTable      ();

    explicit       FrequencyTable      (T const& value);

    void           operator=           (T const& value);

    void           operator()          (T const& value);

    FrequencyTable&
                   operator|=          (FrequencyTable const& other);

    bool           empty               () const;

    size_t         size                () const;

    T const&       mode                () const;

private:

    std::unordered_map<T, size_t> _frequency;

};


/*!
    @brief      Default construct an instance.
*/
template<
    typename T>
inline FrequencyTable<T>::FrequencyTable()

    : _frequency()

{
}


/*!
    @brief      Construct an instance and add @a value.
*/
template<
    typename T>
inline FrequencyTable<T>::FrequencyTable(
    T const& value)

    : _frequency{{value, 1}}

{
}


/*!
    @brief      Assign @a value to the instance.

    This initializes the instance to be reused again.
*/
template<
    typename T>
inline void FrequencyTable<T>::operator=(
    T const& value)
{
    _frequency = {{value, 1}};
}


/*!
    @brief      Add @a value to the instance.

    The layered frequency for @a value is increased by one.
*/
template<
    typename T>
inline void FrequencyTable<T>::operator()(
    T const& value)
{
    ++_frequency[value];
}


/*!
    @brief      Merge @a other with *this.
*/
template<
    typename T>
inline FrequencyTable<T>& FrequencyTable<T>::operator|=(
    FrequencyTable const& other)
{
    for(auto const& pair: other._frequency) {
        auto position = _frequency.find(pair.first);
        if(position == _frequency.end()) {
            _frequency.insert(pair);
        }
        else {
            position->second += pair.second;
        }
    }

    return *this;
}


template<
    typename T>
inline bool FrequencyTable<T>::empty() const
{
    return _frequency.empty();
}


template<
    typename T>
inline size_t FrequencyTable<T>::size() const
{
    return _frequency.size();
}


/*!
    @brief      Return the mode of the histogram.

    In case there are multiple modal values, it is undefined which one is
    returned.
*/
template<
    typename T>
inline T const& FrequencyTable<T>::mode() const
{
    assert(!empty());

    // For speed, _frequency is an unordered_map instead of a regular map.
    // Because of that, we can't make promisses about which value is returned
    // in the multi-modal case. We don't know in which order the values in
    // the map are visited by the algorithm below.

    // Find a value with the highest frequency.
    auto result = std::max_element(_frequency.begin(), _frequency.end(),
        [](auto const& lhs, auto const& rhs) {
            return lhs.second < rhs.second; });

    return result->first;
}


/*!
    @brief      Merge @a lhs with @a rhs.
*/
template<
    typename T>
inline FrequencyTable<T> operator|(
    FrequencyTable<T> const& lhs,
    FrequencyTable<T> const& rhs)
{
    return FrequencyTable<T>(lhs) |= rhs;
}

} // namespace detail
} // namespace accumulator
} // namespace algorithm
} // namespace fern
