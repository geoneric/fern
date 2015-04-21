// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/io/gdal/dataset_handle.h"
#include <cassert>
#include <gdal_priv.h>


namespace fern {
namespace io {
namespace gdal {

/*!
    @brief      Construct a handle based on a GDAL dataset passed in.
*/
DatasetHandle::DatasetHandle(
    GDALDataset* dataset)

    : _dataset(dataset)

{
    assert(_dataset);
}


/*!
    @brief      Destruct the instance, closing the GDAL dataset.
*/
DatasetHandle::~DatasetHandle()
{
    GDALClose(_dataset);
}


/*!
    @brief      Return the layered pointer to the GDAL dataset.
*/
GDALDataset* DatasetHandle::operator->()
{
    return _dataset;
}

} // namespace gdal
} // namespace io
} // namespace fern
