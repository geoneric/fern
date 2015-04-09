// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <array>


namespace fern {

template<
    class Coordinate,
    size_t nr_dimensions>
class Point:
    public std::array<Coordinate, nr_dimensions>
{

    static_assert(nr_dimensions > 0, "");

public:

                   Point               ();

                   Point               (Coordinate const& coordinate);

                   Point               (Coordinate const& coordinate1,
                                        Coordinate const& coordinate2);

                   Point               (Coordinate const& coordinate1,
                                        Coordinate const& coordinate2,
                                        Coordinate const& coordinate3);

private:

};


template<
    class Coordinate,
    size_t nr_dimensions>
inline Point<Coordinate, nr_dimensions>::Point()

    : std::array<Coordinate, nr_dimensions>()

{
}


template<
    class Coordinate,
    size_t nr_dimensions>
inline Point<Coordinate, nr_dimensions>::Point(
    Coordinate const& coordinate)

    : std::array<Coordinate, nr_dimensions>()

{
    std::get<0>(*this) = coordinate;
}


template<
    class Coordinate,
    size_t nr_dimensions>
inline Point<Coordinate, nr_dimensions>::Point(
    Coordinate const& coordinate1,
    Coordinate const& coordinate2)

    : std::array<Coordinate, nr_dimensions>()

{
    std::get<0>(*this) = coordinate1;
    std::get<1>(*this) = coordinate2;
}


template<
    class Coordinate,
    size_t nr_dimensions>
inline Point<Coordinate, nr_dimensions>::Point(
    Coordinate const& coordinate1,
    Coordinate const& coordinate2,
    Coordinate const& coordinate3)

    : std::array<Coordinate, nr_dimensions>()

{
    std::get<0>(*this) = coordinate1;
    std::get<1>(*this) = coordinate2;
    std::get<2>(*this) = coordinate3;
}

} // namespace fern
