#pragma once
#include <type_traits>
#include "fern/feature/core/masked_array.h"


namespace fern {
namespace expression_tree {

//! Wrapper class for rasters.
/*!
*/
template<
    class Result>
struct Raster
{

    // static_assert(!std::is_pointer<Result>::value, "Type must be a class");

    typedef Result value_type;

    typedef Raster<value_type> result_type;

    // typedef fern::MaskedArray<Result, 2> result_type;

    // typedef typename Result::value_type value_type;

    // typedef Result result_type;

    // typedef typename Result::iterator iterator;

    // typedef typename Result::const_iterator const_iterator;

    // typedef typename Result::reference reference;

    // typedef typename Result::const_reference const_reference;

    // typedef typename Result::difference_type difference_type;

    // typedef typename Result::size_type size_type;

    Raster(
        MaskedArray<value_type, 2> const& value)
        : value(value)
    {
    }

    // explicit operator Result const&()
    // {
    //     return value;
    // }

    // const_iterator begin() const
    // {
    //     return std::begin(value);
    // }

    // const_iterator end() const
    // {
    //     return std::end(value);
    // }

    // iterator begin()
    // {
    //     return std::begin(value);
    // }

    // iterator end()
    // {
    //     return std::end(value);
    // }

    // result_type const& value;

    MaskedArray<value_type, 2> value;

};

} // namespace expression_tree
} // namespace fern
