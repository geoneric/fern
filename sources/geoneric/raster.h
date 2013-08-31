#pragma once
#include <algorithm>
#include <array>
#include <boost/range/iterator.hpp>
#include "geoneric/data_traits.h"


namespace geoneric {

template<
    typename ValueType,
    size_t nr_rows_,
    size_t nr_cols_>
class Raster
{
private:

    typedef std::array<ValueType, nr_rows_ * nr_cols_> Array;

    Array            _cells;

public:

    typedef typename Array::iterator iterator;

    typedef typename Array::const_iterator const_iterator;

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
    typedef RasterTag DataCategory;
};

} // namespace geoneric


namespace boost {

template<
    typename ValueType,
    size_t nr_rows,
    size_t nr_cols>
struct range_const_iterator<geoneric::Raster<ValueType, nr_rows, nr_cols>>
{
    typedef typename
        geoneric::Raster<ValueType, nr_rows, nr_cols>::const_iterator type;
};


template<
    typename ValueType,
    size_t nr_rows,
    size_t nr_cols>
struct range_mutable_iterator<geoneric::Raster<ValueType, nr_rows, nr_cols>>
{
    typedef typename
        geoneric::Raster<ValueType, nr_rows, nr_cols>::iterator type;
};

} // namespace boost
