// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <cassert>
#include <netcdf.h>
#include "fern/io/netcdf/core/dataset_handle.h"


namespace fern {
namespace io {
namespace netcdf {
namespace detail {

template<
    typename ValueType>
int                read_0d             (DatasetHandle const& handle,
                                        int variable_id,
                                        ValueType& value);

template<>
inline int read_0d(
    DatasetHandle const& handle,
    int variable_id,
    double& value)
{
    return nc_get_var_double(handle.ncid(), variable_id, &value);
}


template<
    typename OutputNoDataPolicy,
    typename Destination,
    typename DestinationCategory>
void               read_variable       (OutputNoDataPolicy&
                                            output_no_data_policy,
                                        DatasetHandle const& handle,
                                        int variable_id,
                                        Destination& destination,
                                        DestinationCategory const&
                                            destination_category);

template<
    typename OutputNoDataPolicy,
    typename Destination>
void read_variable(
    OutputNoDataPolicy& /* output_no_data_policy */,
    DatasetHandle const& handle,
    int variable_id,
    Destination& destination,
    constant_tag)
{
    int status = read_0d(handle, variable_id, get(destination));

    // TODO Raise low level exception. Assumptions must be met when calling
    //      this function.
    assert(status == NC_NOERR);
}

} // namespace detail
} // namespace netcdf
} // namespace io
} // namespace fern
