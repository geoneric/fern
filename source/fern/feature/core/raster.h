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
#include "fern/feature/core/array.h"


namespace fern {


//! Class for multidimensional rasters, with a cartesian coordinate system.
/*!
    @ingroup    fern_feature_group
    @sa         Array

    Raster extents Array with support for a Cartesian projection
    coordinate system. A raster is an array which is positioned in a
    Cartesian coordinate system.
*/
template<
    typename T,
    size_t nr_dimensions>
class Raster:
    public Array<T, nr_dimensions>
{

public:

    //! Information for transforming cell indices to projection coordinates.
    /*!
        For each dimension, this array contains:
        - Projection coordinate of the border of the first cell.
        - Size of each cell.

        With this information, cell indices can be translated to projection
        coordinates, and vice versa.

        For example, in case of a 2D raster, this array can be used like this:

        \code
        Transformation transformation{{west, cell_width, north, cell_height}};
        \endcode

        In case of a 2D raster, the cell indices of the upper left cell
        are 0, 0. The center of this cell has cell indices 0.5, 0.5.
    */
    using Transformation = std::array<double, 2 * nr_dimensions>;

    template<size_t nr_ranges>
                   Raster              (gen_type<nr_ranges> const& sizes,
                                        Transformation const& transformation);

    template<size_t nr_ranges>
                   Raster              (gen_type<nr_ranges> const& sizes,
                                        Transformation const& transformation,
                                        T const& value);

    Transformation const&
                   transformation      () const;

private:

    Transformation _transformation;

};


template<
    typename T,
    size_t nr_dimensions>
template<
    size_t nr_ranges>
inline Raster<T, nr_dimensions>::Raster(
    gen_type<nr_ranges> const& sizes,
    Transformation const& transformation)

    : Array<T, nr_dimensions>(sizes),
      _transformation(transformation)

{
}


template<
    typename T,
    size_t nr_dimensions>
template<
    size_t nr_ranges>
inline Raster<T, nr_dimensions>::Raster(
    gen_type<nr_ranges> const& sizes,
    Transformation const& transformation,
    T const& value)

    : Array<T, nr_dimensions>(sizes, value),
      _transformation(transformation)

{
}


template<
    typename T,
    size_t nr_dimensions>
inline typename Raster<T, nr_dimensions>::Transformation const&
    Raster<T, nr_dimensions>::transformation() const
{
    return _transformation;
}

} // namespace fern
