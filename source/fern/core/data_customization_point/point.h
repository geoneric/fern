#pragma once
#include "fern/core/data_traits/point.h"


namespace fern {

/// template<
///     class Coordinate,
///     size_t nr_dimensions>
/// struct DataTraits<
///     Point<Coordinate, nr_dimensions>>
/// {
/// 
///     using value_type = Coordinate;
/// 
///     using reference = value_type&;
/// 
///     using const_reference = value_type const&;
/// 
///     static size_t const rank = nr_dimensions;
/// 
/// };


template<
    size_t index,
    class Coordinate,
    size_t nr_dimensions>
inline constexpr typename DataTraits<Point<Coordinate, nr_dimensions>>
        ::const_reference get(
    Point<Coordinate, nr_dimensions> const& point)
{
    return std::get<index>(point);
}


template<
    size_t index,
    class Coordinate,
    size_t nr_dimensions>
inline constexpr typename DataTraits<Point<Coordinate, nr_dimensions>>
        ::reference get(
    Point<Coordinate, nr_dimensions>& point)
{
    return std::get<index>(point);
}

} // namespace fern
