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
#include <vector>
#include "fern/io/netcdf/core/dataset_handle.h"


namespace fern {
namespace io {
namespace netcdf {

/*!
    @ingroup    fern_io_netcdf_group
    @brief      Return the value of the global attribute @a name in @a ncid.

    Assumptions:
    - @a dataset corresponds with a valid open NetCDF dataset.
    - @a name corresponds with a global attribute in the dataset.
*/
template<
    typename T>
T                  attribute           (DatasetHandle const& dataset,
                                        std::string const& name);

bool               has_attribute       (DatasetHandle const& dataset,
                                        std::string const& name);

std::vector<std::string>
                   conventions         (DatasetHandle const& dataset);

} // namespace netcdf
} // namespace io
} // namespace fern
