#pragma once
#include <algorithm>
#include <array>
#include <boost/range/iterator.hpp>
#include "fern/data_traits.h"


namespace fern {

template<
    typename ValueType,
    size_t nr_rows_,
    size_t nr_cols_>
class Raster
{
private:

    using Array = std::array<ValueType, nr_rows_ * nr_cols_>;

    Array            _cells;

public:

    using iterator = typename Array::iterator;

    using const_iterator = typename Array::const_iterator;

    static size_t nr_rows()
    {
        return nr_rows_;
    }

    static size_t nr_cols()
    {
        return nr_cols_;
    }

    Raster()
        : _cells()
    {
    }

    Raster(
        ValueType initial_value)
        : _cells()
    {
        std::fill(_cells.begin(), _cells.end(), initial_value);
    }

    void set(
        size_t row,
        size_t col,
        ValueType value) {
        assert(row < nr_rows_);
        assert(col < nr_cols_);
        _cells[row * nr_rows_ + col] = value;
    }

    ValueType get(
        size_t row,
        size_t col) {
        assert(row < nr_rows_);
        assert(col < nr_cols_);
        return _cells[row * nr_rows_ + col];
    }

    iterator begin()
    {
        return _cells.begin();
    }

    const_iterator begin() const
    {
        return _cells.begin();
    }

    iterator end()
    {
        return _cells.end();
    }

    const_iterator end() const
    {
        return _cells.end();
    }

};


template<
    typename ValueType,
    size_t nr_rows,
    size_t nr_cols>
struct DataTraits<Raster<ValueType, nr_rows, nr_cols>>
{
    using DataCategory = RasterTag;
};

} // namespace fern


namespace boost {

template<
    typename ValueType,
    size_t nr_rows,
    size_t nr_cols>
struct range_const_iterator<fern::Raster<ValueType, nr_rows, nr_cols>>
{
    using type = typename fern::Raster<ValueType, nr_rows, nr_cols>
        ::const_iterator;
};


template<
    typename ValueType,
    size_t nr_rows,
    size_t nr_cols>
struct range_mutable_iterator<fern::Raster<ValueType, nr_rows, nr_cols>>
{
    using type = typename fern::Raster<ValueType, nr_rows, nr_cols>::iterator;
};

} // namespace boost
