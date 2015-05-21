// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/io/fern/hdf5_dataset.h"


namespace fern {
namespace language {

HDF5Dataset::HDF5Dataset()

    : _dataset_id(0)

{
}


HDF5Dataset::HDF5Dataset(
    hid_t id)

    : _dataset_id(id)

{
    assert(_dataset_id > 0);
}



HDF5Dataset::HDF5Dataset(
    HDF5Dataset&& other)

    : _dataset_id(other._dataset_id)

{
    other._dataset_id = 0;
}


HDF5Dataset& HDF5Dataset::operator=(
    HDF5Dataset&& other)
{
    if(_dataset_id > 0) {
        H5Dclose(_dataset_id);
    }

    _dataset_id = other._dataset_id;
    other._dataset_id = 0;

    return *this;
}


HDF5Dataset::~HDF5Dataset()
{
    if(_dataset_id > 0) {
        H5Dclose(_dataset_id);
    }
}


hid_t HDF5Dataset::id() const
{
    assert(_dataset_id > 0);
    return _dataset_id;
}


hid_t HDF5Dataset::type() const
{
    assert(_dataset_id > 0);
    return H5Dget_type(_dataset_id);
}


H5T_class_t HDF5Dataset::type_class() const
{
    assert(_dataset_id > 0);
    return H5Tget_class(type());
}


hid_t HDF5Dataset::space() const
{
    assert(_dataset_id > 0);

    H5D_space_status_t info;
    herr_t status = H5Dget_space_status(_dataset_id, &info);
    assert(status == 0);
    assert(info == H5D_SPACE_STATUS_ALLOCATED);

    hid_t id = H5Dget_space(_dataset_id);
    assert(id > 0);
    return id;
}


bool HDF5Dataset::space_is_simple() const
{
    assert(_dataset_id > 0);

    hid_t space = this->space();
    bool is_simple = H5Sis_simple(space);

    herr_t status = H5Sclose(space);
    assert(status == 0);

    return is_simple;
}


H5S_class_t HDF5Dataset::extent_type() const
{
    assert(_dataset_id > 0);
    assert(space_is_simple());

    hid_t space = this->space();
    H5S_class_t type = H5Sget_simple_extent_type(space);
    assert(type >= 0);

    herr_t status = H5Sclose(space);
    assert(status == 0);

    return type;
}

} // namespace language
} // namespace fern
