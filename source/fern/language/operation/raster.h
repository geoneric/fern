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
#include <array>
#include <cassert>
#include <boost/range/iterator.hpp>
#include "fern/language/operation/data_type_traits.h"


namespace fern {
namespace language {

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
struct DataTypeTraits<Raster<ValueType, nr_rows, nr_cols>>
{
    using DataCategory = RasterTag;
};

} // namespace language
} // namespace fern


namespace boost {

template<
    typename ValueType,
    size_t nr_rows,
    size_t nr_cols>
struct range_const_iterator<fern::language::Raster<ValueType, nr_rows, nr_cols>>
{
    using type = typename fern::language::Raster<ValueType, nr_rows, nr_cols>
        ::const_iterator;
};


template<
    typename ValueType,
    size_t nr_rows,
    size_t nr_cols>
struct range_mutable_iterator<fern::language::Raster<ValueType, nr_rows, nr_cols>>
{
    using type = typename fern::language::Raster<ValueType, nr_rows, nr_cols>::iterator;
};

} // namespace boost
