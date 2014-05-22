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

    using value_type = Result;

    using result_type = Raster<value_type>;

    // using result_type = fern::MaskedArray<Result, 2>;

    // using value_type = typename Result::value_type;

    // using result_type = Result;

    // using iterator = typename Result::iterator;

    // using const_iterator = typename Result::const_iterator;

    // using reference = typename Result::reference;

    // using const_reference = typename Result::const_reference;

    // using difference_type = typename Result::difference_type;

    // using size_type = typename Result::size_type;

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
