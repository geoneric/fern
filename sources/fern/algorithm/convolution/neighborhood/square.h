#pragma once
#include <algorithm>
#include <cassert>


namespace fern {

template<
    class T,
    size_t radius>
class Square
{

public:

                   Square              (std::initializer_list<
                                           std::initializer_list<T>> const&
                                               weights);

    T const*       operator[]          (size_t index) const;

    static constexpr size_t  size      ();

private:

    static size_t const _size = 2 * radius + 1;

    T              _weights[_size][_size];

};


template<
    class T,
    size_t radius>
inline constexpr size_t  Square<T, radius>::size()
{
    return _size;
}


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
        std::copy(it->begin(), it->end(), _weights[i]);
        ++it;
    }
}


template<
    class T,
    size_t radius>
inline T const* Square<T, radius>::operator[](
    size_t index) const
{
    assert(index < _size);
    return _weights[index];
}

} // namespace fern
