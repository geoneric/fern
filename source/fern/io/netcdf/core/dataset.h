// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <string>
#include <netcdf.h>
#include "fern/io/netcdf/core/dataset_handle.h"


namespace fern {
namespace io {
namespace netcdf {

DatasetHandle      open_dataset        (std::string const& name,
                                        int mode=NC_NOWRITE);

} // namespace netcdf
} // namespace io
} // namespace fern
