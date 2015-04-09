// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <memory>
#include <tuple>


namespace fern {
namespace python {
namespace detail {

template<
    typename T>
class MaskedRaster
{

public:

                   MaskedRaster        (std::tuple<size_t, size_t> sizes,
                                        std::tuple<double, double> origin,
                                        std::tuple<double, double> cell_sizes);

                   MaskedRaster        (std::tuple<size_t, size_t> sizes,
                                        std::tuple<double, double> origin,
                                        std::tuple<double, double> cell_sizes,
                                        T const& value);

                   MaskedRaster        (MaskedRaster const& other);

                   ~MaskedRaster       ()=default;

    std::tuple<size_t, size_t> const&
                   sizes               () const;

    size_t         size                () const;

    std::tuple<double, double> const&
                   origin              () const;

    std::tuple<double, double> const&
                   cell_sizes          () const;

    double         cell_area           () const;

    T*             data                ();

    T const*       data                () const;

    size_t         index               (size_t index1,
                                        size_t index2) const;

    T&             element             (size_t index);

    T const&       element             (size_t index) const;

    T&             element             (size_t index1,
                                        size_t index2);

    T const&       element             (size_t index1,
                                        size_t index2) const;

private:

    std::tuple<size_t, size_t> _sizes;

    std::tuple<double, double> _origin;

    std::tuple<double, double> _cell_sizes;

    std::unique_ptr<T[]> _data;

};


template<
    typename T>
inline MaskedRaster<T>::MaskedRaster(
    std::tuple<size_t, size_t> sizes,
    std::tuple<double, double> origin,
    std::tuple<double, double> cell_sizes)

    : _sizes(sizes),
      _origin(origin),
      _cell_sizes(cell_sizes),
      _data(new T[std::get<0>(sizes) * std::get<1>(sizes)])

{
}


template<
    typename T>
inline MaskedRaster<T>::MaskedRaster(
    std::tuple<size_t, size_t> sizes,
    std::tuple<double, double> origin,
    std::tuple<double, double> cell_sizes,
    T const& value)

    : MaskedRaster(sizes, origin, cell_sizes)

{
    std::fill(_data.get(), _data.get() + size(), value);
}


template<
    typename T>
inline MaskedRaster<T>::MaskedRaster(
    MaskedRaster const& other)

    : MaskedRaster(other.sizes(), other.origin(), other.cell_sizes())

{
    std::copy(other._data.get(), other._data.get() + size(), _data.get());
}


template<
    typename T>
inline size_t MaskedRaster<T>::size() const
{
    return std::get<0>(_sizes) * std::get<1>(_sizes);
}


template<
    typename T>
inline std::tuple<size_t, size_t> const& MaskedRaster<T>::sizes() const
{
    return _sizes;
}


template<
    typename T>
inline std::tuple<double, double> const& MaskedRaster<T>::origin() const
{
    return _origin;
}


template<
    typename T>
inline std::tuple<double, double> const& MaskedRaster<T>::cell_sizes() const
{
    return _cell_sizes;
}


template<
    typename T>
inline double MaskedRaster<T>::cell_area() const
{
    return std::get<0>(_cell_sizes) * std::get<1>(_cell_sizes);
}


template<
    typename T>
inline T* MaskedRaster<T>::data()
{
    return _data.get();
}


template<
    typename T>
inline T const* MaskedRaster<T>::data() const
{
    return _data.get();
}


template<
    typename T>
inline size_t MaskedRaster<T>::index(
    size_t index1,
    size_t index2) const
{
    return index1 * std::get<1>(_sizes) + index2;
}


template<
    typename T>
inline T& MaskedRaster<T>::element(
    size_t index)
{
    return _data[index];
}


template<
    typename T>
inline T const& MaskedRaster<T>::element(
    size_t index) const
{
    return _data[index];
}


template<
    typename T>
inline T& MaskedRaster<T>::element(
    size_t index1,
    size_t index2)
{
    return _data[index(index1, index2)];
}


template<
    typename T>
inline T const& MaskedRaster<T>::element(
    size_t index1,
    size_t index2) const
{
    return _data[index(index1, index2)];
}

} // namespace detail
} // namespace python
} // namespace fern
