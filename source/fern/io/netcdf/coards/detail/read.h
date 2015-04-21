// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <algorithm>
#include <netcdf.h>
#include "fern/core/io_error.h"
#include "fern/core/type_traits.h"
#include "fern/io/core/file.h"
#include "fern/io/netcdf/core/attribute.h"
#include "fern/io/netcdf/core/dataset.h"
#include "fern/io/netcdf/core/variable.h"
#include "fern/io/netcdf/coards/dataset.h"


namespace fern {
namespace io {
namespace netcdf {
namespace detail {

template<
    typename OutputNoDataPolicy,
    typename Destination,
    typename DestinationCategory
>
void               read_coards         (OutputNoDataPolicy&
                                            output_no_data_policy,
                                        DataName const& data_name,
                                        Destination& destination,
                                        DestinationCategory const&
                                            destination_category);


template<
    typename OutputNoDataPolicy,
    typename Destination
>
inline void read_coards(
    OutputNoDataPolicy& output_no_data_policy,
    DataName const& data_name,
    Destination& destination,
    constant_tag)
{
    // Read a scalar value from the Coards dataset pointed to by data_name.


    // Try to open the NetCDF file.
    std::string pathname{data_name.database_pathname().native_string()};

    if(!file_exists(pathname)) {
        throw IOError(pathname,
            Exception::messages()[MessageId::DOES_NOT_EXIST]);
    }

    auto handle = open_dataset(pathname);


    // Verify that the file is formatted according to the COARDS conventions.
    if(!conforms_to_coards(handle)) {
        throw IOError(pathname,
            Exception::messages().format_message(
                MessageId::DOES_NOT_CONFORM_TO_CONVENTION, "COARDS"));
    }


    // Verify that the requested variable is in the dataset.
    assert(data_name.data_pathname().names().size() == 1u);
    std::string variable_name{data_name.data_pathname().names()[0]};
    if(!contains_variable(handle, variable_name)) {
        throw IOError(pathname,
            Exception::messages().format_message(
                MessageId::DOES_NOT_CONTAIN_VARIABLE, variable_name));
    }


    // Verify that the requested variable is a scalar.
    int variable_id = netcdf::variable_id(handle, variable_name);
    if(!variable_is_scalar(handle, variable_id)) {
        throw IOError(pathname,
            Exception::messages().format_message(
                MessageId::VARIABLE_IS_NOT_A_SCALAR, variable_name));
    }


    // Verify that the requested variable has the same value type as the
    // variable passed in.
    ValueType value_type = netcdf::value_type_id(handle, variable_id);
    assert(fern::value_type_id<fern::value_type<Destination>>() == value_type);


    // Read the variable.
    read_variable(output_no_data_policy, handle, variable_id, destination);
}

} // namespace detail
} // namespace netcdf
} // namespace io
} // namespace fern
