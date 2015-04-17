// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/io/netcdf/core/variable.h"
#include <cassert>
#include <netcdf.h>
#include <map>


namespace fern {
namespace io {
namespace netcdf {

/*!
    @ingroup    fern_io_netcdf_group
    @brief      Return whether or not dataset \a handle contains variable
                @a name.

    Assumptions:
    - @a handle corresponds with a valid open NetCDF dataset.
*/
bool contains_variable(
    DatasetHandle const& handle,
    std::string const& name)
{
    int variable_id;
    int status = nc_inq_varid(handle.ncid(), name.c_str(), &variable_id);

    assert(status != NC_EBADID);

    return status == NC_NOERR;
}


/*!
    @ingroup    fern_io_netcdf_group
    @brief      Return id of variable @a name.

    Assumptions:
    - @a handle corresponds with a valid open NetCDF dataset.
    - @a name corresponds with a variable in the dataset.
*/
int variable_id(
    DatasetHandle const& handle,
    std::string const& name)
{
    int variable_id;
    int status = nc_inq_varid(handle.ncid(), name.c_str(), &variable_id);

    assert(status != NC_EBADID);
    assert(status == NC_NOERR);

    return variable_id;
}


/*!
    @ingroup    fern_io_netcdf_group
    @brief      Return whether variable @a variable_id is scalar.

    Assumptions:
    - @a handle corresponds with a valid open NetCDF dataset.
    - @a variable_id corresponds with the id of a variable in the dataset.
*/
bool variable_is_scalar(
    DatasetHandle const& handle,
    int variable_id)
{
    int nr_dimensions;
    int status = nc_inq_var(handle.ncid(), variable_id, nullptr, nullptr,
        &nr_dimensions, nullptr, nullptr);

    assert(status != NC_EBADID);
    assert(status != NC_ENOTVAR);
    assert(status == NC_NOERR);
    assert(nr_dimensions >= 0);

    return nr_dimensions == 0;
}


static std::map<int, ValueType> value_type_map {

    // NC_NAT: Not A Type.

    {NC_CHAR, VT_CHAR},

    {NC_UBYTE, VT_UINT8},
    {NC_BYTE, VT_INT8},

    {NC_USHORT, VT_UINT16},
    {NC_SHORT, VT_INT16},

    {NC_UINT, VT_UINT32},
    {NC_INT, VT_INT32},
    {NC_LONG, VT_INT32},

    {NC_UINT64, VT_UINT64},
    {NC_INT64, VT_INT64},

    {NC_FLOAT, VT_FLOAT32},
    {NC_DOUBLE, VT_FLOAT64},

    {NC_STRING, VT_STRING}
};


ValueType value_type_id(
    nc_type type_id)
{
    auto value_type_it = value_type_map.find(type_id);
    assert(value_type_it != value_type_map.end());

    return value_type_it->second;
}


/*!
    @ingroup    fern_io_netcdf_group
    @brief      Return value type id of variable @a variable_id.

    Assumptions:
    - @a handle corresponds with a valid open NetCDF dataset.
    - @a variable_id corresponds with the id of a variable in the dataset.
*/
ValueType value_type_id(
    DatasetHandle const& handle,
    int variable_id)
{
    nc_type type_id;
    int status = nc_inq_vartype(handle.ncid(), variable_id, &type_id);

    assert(status != NC_EBADID);
    assert(status != NC_ENOTVAR);
    assert(status == NC_NOERR);

    return value_type_id(type_id);
}

} // namespace netcdf
} // namespace io
} // namespace fern
