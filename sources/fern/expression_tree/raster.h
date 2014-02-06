#pragma once
#include <type_traits>


namespace fern {

//! Wrapper class for rasters.
/*!
*/
template<
    class Result>
struct Raster
{

    static_assert(!std::is_pointer<Result>::value, "Type must be a class");

    typedef typename Result::value_type value_type;

    typedef Result result_type;

    // typedef typename Result::iterator iterator;

    // typedef typename Result::const_iterator const_iterator;

    // typedef typename Result::reference reference;

    // typedef typename Result::const_reference const_reference;

    // typedef typename Result::difference_type difference_type;

    // typedef typename Result::size_type size_type;

    Raster(
        Result const& value)
        : value(value)
    {
    }

    explicit operator Result const&()
    {
        return value;
    }

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

    Result const& value;

};

} // namespace fern
