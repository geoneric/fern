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
#include <cassert>


namespace fern {

//! Class for square neighborhoods.
/*!
  \tparam    T Type of values to store in the neighborhood.
  \tparam    radius Radius of the neighborhood. A radius of 1 results in a
             3x3 square. Square neighborhoods always have an odd size.
*/
template<
    class T,
    size_t radius>
class Square
{

public:

    using value_type = T;

    using reference = T&;

    using const_reference = T const&;

                   Square              (std::initializer_list<
                                           std::initializer_list<T>> const&
                                               weights);

    T&                weight              (size_t index);

    T const&          weight              (size_t index) const;

    static constexpr size_t  size      ();

private:

    static size_t const _size = 2 * radius + 1;

    T              _weights[_size * _size];

};


//! Return the size of the square.
/*!
  The size is the length of each side of the square. The number returned is
  always odd.
*/
template<
    class T,
    size_t radius>
inline constexpr size_t  Square<T, radius>::size()
{
    return _size;
}


//! Construct a square given \a weights.
/*!
  \param     weights Weights to store in the square.
  \warning   \a weights must have the same number of rows as columns. This
             number must be equal to 2 * \a radius + 1.
*/
template<
    class T,
    size_t radius>
inline Square<T, radius>::Square(
    std::initializer_list<std::initializer_list<T>> const& weights)
{
    auto it = weights.begin();

    assert(weights.size() == _size);
    for(size_t i = 0; i < _size; ++i) {
        assert(it->size() == _size);
        T* target = &_weights[i * _size];

        // We are passing an array as the target of the copy. By default,
        // Visual Studio throws a C4996 warning because of that, which we
        // silence here.
        std::copy(it->begin(), it->end(),
#if defined(_MSC_VER)
            stdext::make_unchecked_array_iterator(target)
#else
            target
#endif
        );
        ++it;
    }
}


template<
    class T,
    size_t radius>
inline T& Square<T, radius>::weight(
    size_t index)
{
    assert(index < _size * _size);
    return _weights[index];
}


template<
    class T,
    size_t radius>
inline T const& Square<T, radius>::weight(
    size_t index) const
{
    assert(index < _size * _size);
    return _weights[index];
}

} // namespace fern
