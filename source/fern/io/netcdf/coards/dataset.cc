// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/io/netcdf/coards/dataset.h"
#include <algorithm>
#include "fern/io/netcdf/core/attribute.h"


namespace fern {
namespace io {
namespace netcdf {

/*!
    @ingroup    fern_io_netcdf_group
    @brief      Return whether dataset @a handle conforms to the COARDS
                conventions.

    Assumptions:
    - @a handle corresponds with a valid open NetCDF dataset.
*/
bool conforms_to_coards(
    DatasetHandle const& handle)
{
    auto conventions(netcdf::conventions(handle));

    return std::find(conventions.begin(), conventions.end(), "COARDS") !=
            conventions.end();
}

} // namespace netcdf
} // namespace io
} // namespace fern
