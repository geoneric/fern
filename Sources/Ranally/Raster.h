#pragma once
#include <algorithm>
#include <array>
#include "Ranally/DataTraits.h"


namespace ranally {

template<
    typename ValueType,
    size_t nrRows_,
    size_t nrCols_>
class Raster
{
private:

    typedef std::array<ValueType, nrRows_ * nrCols_> Array;

    Array            _cells;

public:

    typedef typename Array::iterator iterator;

    typedef typename Array::const_iterator const_iterator;

    static size_t nrRows()
    {
        return nrRows_;
    }

    static size_t nrCols()
    {
        return nrCols_;
    }

    Raster()
        : _cells()
    {
    }

    Raster(
        ValueType initialValue)
        : _cells()
    {
        std::fill(_cells.begin(), _cells.end(), initialValue);
    }

    void set(
        size_t row,
        size_t col,
        ValueType value) {
        assert(row < nrRows_);
        assert(col < nrCols_);
        _cells[row * nrRows_ + col] = value;
    }

    ValueType get(
        size_t row,
        size_t col) {
        assert(row < nrRows_);
        assert(col < nrCols_);
        return _cells[row * nrRows_ + col];
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
    size_t nrRows,
    size_t nrCols>
struct DataTraits<Raster<ValueType, nrRows, nrCols> >
{
    typedef RasterTag DataCategory;
};

} // namespace ranally


namespace boost {

template<
    typename ValueType,
    size_t nrRows,
    size_t nrCols>
struct range_const_iterator<ranally::Raster<ValueType, nrRows, nrCols> >
{
    typedef typename ranally::Raster<ValueType, nrRows, nrCols>::const_iterator
        type;
};


template<
    typename ValueType,
    size_t nrRows,
    size_t nrCols>
struct range_mutable_iterator<ranally::Raster<ValueType, nrRows, nrCols> >
{
    typedef typename ranally::Raster<ValueType, nrRows, nrCols>::iterator type;
};

} // namespace boost
