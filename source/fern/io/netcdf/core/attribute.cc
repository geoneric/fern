// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/io/netcdf/core/attribute.h"
#include <cassert>
#include <netcdf.h>
#include "fern/core/string.h"


namespace fern {
namespace io {
namespace netcdf {

template<>
std::string attribute(
    DatasetHandle const& dataset,
    std::string const& name)
{
    assert(has_attribute(dataset, name));

    size_t buffer_size;
    int status = nc_inq_attlen(dataset.ncid(), NC_GLOBAL, name.c_str(),
        &buffer_size);

    assert(status == NC_NOERR);

    std::vector<char> buffer(buffer_size);
    status = nc_get_att_text(dataset.ncid(), NC_GLOBAL, name.c_str(),
        buffer.data());

    assert(status == NC_NOERR);
    assert(buffer.empty() || buffer.back() != '\0');

    return std::string(buffer.begin(), buffer.end());
}


/*!
    @ingroup    fern_io_netcdf_group
    @brief      Return whether or not file @a ncid contains global attribute
                @a name.

    Assumptions:
    - @a dataset corresponds with a valid open NetCDF dataset.
*/
bool has_attribute(
    DatasetHandle const& dataset,
    std::string const& name)
{
    int attribute_id;
    int status = nc_inq_attid(dataset.ncid(), NC_GLOBAL, name.c_str(),
        &attribute_id);

    assert(status != NC_EBADID);
    return status == NC_NOERR;
}


/*!
    @ingroup    fern_io_netcdf_group
    @brief      Return the value of the global attribute 'Conventions' as a
                collection of strings.

    If the dataset does not contain an global attribute called 'Convensions',
    then an empty collection is returned.

    The value of the conventions attribute is a whitespace or comma
    separated list of convention names. If a comma is found in the attribute
    value, then the value is split by comma's, which makes it possible for
    convention names to contain whitespace. Otherwise the value is split
    by whitespace.

    Assumptions:
    - @a dataset corresponds with a valid open NetCDF dataset.
*/
std::vector<std::string> conventions(
    DatasetHandle const& dataset)
{
    std::vector<std::string> values;

    if(has_attribute(dataset, "Conventions")) {

        // Get the attribute value as a string.
        std::string value{attribute<std::string>(dataset, "Conventions")};

        // If the list of conventions is seperated by comma's.
        if(value.find(",") != std::string::npos) {
            // Split the list of conventions at the comma's.
            values = split(value, ",");

            // Strip all surrounding whitespace from the elements.
            for(auto& value: values) {
                strip(value);
            }
        }
        else {
            // Split the list of conventions at the whitespace.
            values = split(value);
        }

    }

    return values;
}

} // namespace netcdf
} // namespace io
} // namespace fern
