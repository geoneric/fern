// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/io/netcdf/core/dataset.h"
#include "fern/core/io_error.h"


namespace fern {
namespace io {
namespace netcdf {

/*!
    @ingroup    fern_io_netcdf_group
    @brief      Open dataset @a name and return the handle.
    @throws     IOError In case @a name cannot be opened.
*/
DatasetHandle open_dataset(
    std::string const& name,
    int mode)
{
    int ncid;
    int status = nc_open(name.c_str(), mode, &ncid);

    if(status != NC_NOERR) {
        throw IOError(name,
            Exception::messages()[MessageId::CANNOT_BE_READ]);
    }

    return DatasetHandle(ncid);
}

} // namespace netcdf
} // namespace io
} // namespace fern
