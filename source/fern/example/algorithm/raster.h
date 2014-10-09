#pragma once
#include <cstddef>
#include <vector>


namespace example {

template<
    typename T>
class Raster
{

public:

                   Raster              ();

                   Raster              (Raster&& other);

                   Raster              (double cell_size,
                                        size_t nr_rows,
                                        size_t nr_cols);

    Raster<T>&     operator=           (Raster<T>&& other);

    std::vector<T>& values             ();

    std::vector<T> const&
                    values             () const;

    double          cell_size          () const;

    size_t          nr_rows            () const;

    size_t          nr_cols            () const;

private:

    double          _cell_size;

    size_t          _nr_rows;

    size_t          _nr_cols;

    std::vector<T> _values;

};


template<
    typename T>
Raster<T>::Raster()

    : _cell_size(),
      _nr_rows(),
      _nr_cols(),
      _values()

{
}


template<
    typename T>
Raster<T>::Raster(Raster&& other)

    : _cell_size(other._cell_size),
      _nr_rows(other._nr_rows),
      _nr_cols(other._nr_cols),
      _values(std::move(other._values))

{
}


template<
    typename T>
Raster<T>::Raster(
    double cell_size,
    size_t nr_rows,
    size_t nr_cols)

    : _cell_size(cell_size),
      _nr_rows(nr_rows),
      _nr_cols(nr_cols),
      _values(nr_rows * nr_cols)

{
}


template<
    typename T>
Raster<T>& Raster<T>::operator=(
    Raster<T>&& other)
{
    _cell_size = other._cell_size;
    _nr_rows = other._nr_rows;
    _nr_cols = other._nr_cols;
    _values = std::move(other._values);

    return *this;
}


template<
    typename T>
std::vector<T>& Raster<T>::values()
{
    return _values;
}


template<
    typename T>
std::vector<T> const& Raster<T>::values() const
{
    return _values;
}


template<
    typename T>
double Raster<T>::cell_size() const
{
    return _cell_size;
}


template<
    typename T>
size_t Raster<T>::nr_rows() const
{
    return _nr_rows;
}


template<
    typename T>
size_t Raster<T>::nr_cols() const
{
    return _nr_cols;
}

} // namespace example
