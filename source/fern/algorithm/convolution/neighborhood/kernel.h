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
#include <memory>


namespace fern {

template<
    class T>
class Kernel
{

public:

    using value_type = T;

    using reference = T&;

    using const_reference = T const&;

                   Kernel              (std::size_t radius,
                                        T weight);

                   Kernel              (std::size_t radius,
                                        std::initializer_list<
                                           std::initializer_list<T>> const&
                                               weights);

                   Kernel              (Kernel const&)=default;

                   Kernel              (Kernel&&)=default;

                   ~Kernel             ()=default;

    Kernel&        operator=           (Kernel const&)=default;

    Kernel&        operator=           (Kernel&&)=default;

    std::size_t    radius              () const;

    std::size_t    size                () const;

    T const&       weight              (std::size_t index) const;

    T&             weight              (std::size_t index);

private:

    std::size_t const _radius;

    std::size_t const _size;

    // Pointer instead of vector to make a kernel of boolean weights work
    std::unique_ptr<T[]> _weights;

};


template<
    typename T>
inline Kernel<T>::Kernel(
    std::size_t const radius,
    T const weight)

    : _radius{radius},
      _size{2 * _radius + 1},
      _weights{std::make_unique<T[]>(_size * _size)}

{
    std::fill(_weights, _weights + _size * _size, weight);
}


template<
    typename T>
inline Kernel<T>::Kernel(
    std::size_t const radius,
    std::initializer_list<std::initializer_list<T>> const& weights)

    : _radius{radius},
      _size{2 * _radius + 1},
      _weights{std::make_unique<T[]>(_size * _size)}

{
    auto source_it = weights.begin();  // 2D list
    auto target_it = _weights.get();  // 1D array

    assert(weights.size() == _size);

    for(std::size_t i = 0; i < _size; ++i) {
        assert(source_it->size() == _size);

        std::copy(source_it->begin(), source_it->end(), target_it);

        ++source_it;
        target_it += _size;
    }
}


template<
    typename T>
inline std::size_t Kernel<T>::radius() const
{
    return _radius;
}


//! Return the size of the square.
/*!
  The size is the length of each side of the square. The number returned is
  always odd.
*/
template<
    typename T>
inline std::size_t Kernel<T>::size() const
{
    return _size;
}


template<
    typename T>
inline T const& Kernel<T>::weight(
    std::size_t const index) const
{
    assert(index < _size * _size);
    return _weights[index];
}


template<
    typename T>
inline T& Kernel<T>::weight(
    std::size_t const index)
{
    assert(index < _size * _size);
    return _weights[index];
}

} // namespace fern
