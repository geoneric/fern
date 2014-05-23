#pragma once
#include <cstddef>
#include <vector>


namespace example {

template<
    class T>
class Raster
{

public:

                   Raster              (size_t nr_rows,
                                        size_t nr_cols);

    std::vector<T>& values             ();

    std::vector<T> const&
                    values             () const;

    size_t          nr_rows            () const;

    size_t          nr_cols            () const;

private:

    size_t const   _nr_rows;

    size_t const   _nr_cols;

    std::vector<T> _values;

};


template<
    class T>
Raster<T>::Raster(
    size_t nr_rows,
    size_t nr_cols)

    : _nr_rows(nr_rows),
      _nr_cols(nr_cols),
      _values(nr_rows * nr_cols)

{
}


template<
    class T>
std::vector<T>& Raster<T>::values()
{
    return _values;
}


template<
    class T>
std::vector<T> const& Raster<T>::values() const
{
    return _values;
}


template<
    class T>
size_t Raster<T>::nr_rows() const
{
    return _nr_rows;
}


template<
    class T>
size_t Raster<T>::nr_cols() const
{
    return _nr_cols;
}

} // namespace example
