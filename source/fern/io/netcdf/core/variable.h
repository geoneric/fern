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
#include "fern/core/data_traits.h"
#include "fern/core/value_type.h"
#include "fern/io/netcdf/core/detail/variable.h"


namespace fern {
namespace io {
namespace netcdf {

bool               contains_variable   (DatasetHandle const& handle,
                                        std::string const& name);

int                variable_id         (DatasetHandle const& handle,
                                        std::string const& name);

bool               variable_is_scalar  (DatasetHandle const& handle,
                                        int variable_id);

ValueType          value_type_id       (DatasetHandle const& handle,
                                        int variable_id);

/*!
    @ingroup    fern_io_netcdf_group
    @brief      Read variable @a variable_id from dataset @a handle into
                @a destination.

    Assumptions:
    - @a handle corresponds with a valid open NetCDF dataset.
    - @a variable_id corresponds with the id of a variable in the dataset.
    - @a destination has the ѕame data type as the variable to read.
*/
template<
    typename OutputNoDataPolicy,
    typename Destination>
inline void read_variable(
    OutputNoDataPolicy& output_no_data_policy,
    DatasetHandle const& handle,
    int variable_id,
    Destination& destination)
{
    detail::read_variable(output_no_data_policy, handle, variable_id,
        destination, argument_category<Destination>());
}

} // namespace netcdf
} // namespace io
} // namespace fern
