// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <gdal_priv.h>
#include "fern/core/data_name.h"
#include "fern/io/gdal/dataset_handle.h"


namespace fern {
namespace io {
namespace gdal {

DatasetHandle      open_dataset        (std::string const& name,
                                        GDALAccess mode);

DatasetHandle      open_dataset        (DataName const& data_name,
                                        GDALAccess mode);

GDALRasterBand*    raster_band         (DatasetHandle& dataset,
                                        int band_id);

GDALDataType       gdal_value_type     (DataName const& data_name);

} // namespace gdal
} // namespace io
} // namespace fern
