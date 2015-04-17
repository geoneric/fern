// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/io/netcdf/core/dataset_handle.h"
#include <netcdf.h>


namespace fern {
namespace io {
namespace netcdf {

/*!
    @brief      Construct a handle based on a NetCDF dataset id passed in.
*/
DatasetHandle::DatasetHandle(
    int ncid)

    : _ncid(ncid)

{
}


/*!
    @brief      Destruct the instance, closing the NetCDF dataset.
*/
DatasetHandle::~DatasetHandle()
{
    static_cast<void>(nc_close(_ncid));
}


/*!
    @brief      Return the integer corresponding with the NetCDF dataset id.
*/
int DatasetHandle::ncid() const
{
    return _ncid;
}

} // namespace netcdf
} // namespace io
} // namespace fern
