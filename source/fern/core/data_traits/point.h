// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <cstddef>
#include "fern/core/data_traits.h"
#include "fern/core/point.h"


namespace fern {

template<
    class Coordinate,
    size_t nr_dimensions>
struct DataTraits<
    Point<Coordinate, nr_dimensions>>
{

    using value_type = Coordinate;

    using reference = value_type&;

    using const_reference = value_type const&;

    static size_t const rank = nr_dimensions;

};


/// template<
///     size_t index,
///     class Coordinate,
///     size_t nr_dimensions>
/// inline constexpr typename DataTraits<Point<Coordinate, nr_dimensions>>
///         ::const_reference get(
///     Point<Coordinate, nr_dimensions> const& point)
/// {
///     return std::get<index>(point);
/// }
/// 
/// 
/// template<
///     size_t index,
///     class Coordinate,
///     size_t nr_dimensions>
/// inline constexpr typename DataTraits<Point<Coordinate, nr_dimensions>>
///         ::reference get(
///     Point<Coordinate, nr_dimensions>& point)
/// {
///     return std::get<index>(point);
/// }

} // namespace fern
