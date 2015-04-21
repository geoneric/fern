// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <type_traits>
#include "fern/core/data_name.h"
#include "fern/core/data_traits.h"
#include "detail/read.h"


namespace fern {
namespace io {
namespace gdal {

/*!
    @ingroup    fern_io_gdal_group
    @brief      Read information from a GDAL dataset @a source into
                @a destination.
    @throws     IOError If @a source cannot be opened for reading.
    @throws     IOError If @a source does not contain the requested
                information.
    @throws     IOError If the data type of the information and
                @a destination are not the same.

    This function reads GDAL raster values from a @a source into a @a
    destination. The properties of the @a destination determine which
    values will be read from the @a source. Currently, this function
    support reading whole raster bands only. So, the @a destination must
    be large enough to contain all values from one raster band. Also, the
    transformation information of the @a destination is not updated. It
    is assumed that it already is set to the correct values for the
    whole raster band.

    In the future, this function can support the read of a selection of
    a raster band (a region). This selection can then be configured by
    the size of the @a destination and the transformation.
*/
template<
    typename OutputNoDataPolicy,
    typename Source,
    typename Destination>
void               read                (OutputNoDataPolicy const&
                                            output_no_data_policy,
                                        Source const& source,
                                        Destination& destination);


template<
    typename OutputNoDataPolicy,
    typename Destination
>
inline void read(
    OutputNoDataPolicy& output_no_data_policy,
    DataName const& data_name,
    Destination& destination)
{
    static_assert(std::is_same<argument_category<Destination>,
        raster_2d_tag>::value, "");
    detail::read(output_no_data_policy, data_name, destination);
}

} // namespace gdal
} // namespace io
} // namespace fern
