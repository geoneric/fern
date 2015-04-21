// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/core/data_name.h"
#include "fern/core/data_traits.h"
#include "detail/read.h"


namespace fern {
namespace io {
namespace netcdf {

/*!
    @ingroup    fern_io_netcdf_group
    @brief      Read information from a NetCDF dataset @a source into
                @a destination.
    @throws     IOError If @a source cannot be opened for reading.
    @throws     IOError If @a source does not conform to the COARDS
                convention.
    @throws     IOError If @a source does not contain the requested
                information.
    @throws     IOError If the data type of the information and
                @a destination are not the same.
*/
template<
    typename OutputNoDataPolicy,
    typename Source,
    typename Destination
>
void               read_coards         (OutputNoDataPolicy&
                                            output_no_data_policy,
                                        Source const& source,
                                        Destination& destination);


template<
    typename OutputNoDataPolicy,
    typename Destination
>
inline void read_coards(
    OutputNoDataPolicy& output_no_data_policy,
    DataName const& data_name,
    Destination& destination)
{
    detail::read_coards(output_no_data_policy, data_name, destination,
        argument_category<Destination>());
}

} // namespace netcdf
} // namespace io
} // namespace fern
