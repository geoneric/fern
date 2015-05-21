// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/io/fern/hdf5_file.h"
#include <cassert>


namespace fern {
namespace language {

HDF5File::HDF5File()

    : _file_id(0)

{
}


HDF5File::HDF5File(
    hid_t id)

    : _file_id(id)

{
    assert(_file_id >= 0);
}


HDF5File::~HDF5File()
{
    if(_file_id > 0) {
        H5Fclose(_file_id);
    }
}


hid_t HDF5File::id() const
{
    assert(_file_id > 0);
    return _file_id;
}


hid_t HDF5File::open_dataset(
    std::string const& pathname) const
{
    assert(_file_id > 0);

    hid_t dataset_id = H5Dopen(_file_id, pathname.c_str(), H5P_DEFAULT);
    assert(dataset_id > 0);

    return dataset_id;
}


hid_t HDF5File::open_group(
    std::string const& pathname) const
{
    assert(_file_id > 0);
    return H5Gopen(_file_id, pathname.c_str(), H5P_DEFAULT);
}


bool HDF5File::is_group(
    std::string const& pathname) const
{
    assert(_file_id > 0);
    bool result = false;

    if(H5Lexists(_file_id, pathname.c_str(), H5P_DEFAULT)) {
        H5G_stat_t info;

        herr_t status = H5Gget_objinfo(_file_id, pathname.c_str(), 0, &info);
        assert(status == 0);

        result = info.type == H5G_GROUP;
    }

    return result;

}


bool HDF5File::is_dataset(
    std::string const& pathname) const
{
    assert(_file_id > 0);
    bool result = false;

    if(H5Lexists(_file_id, pathname.c_str(), H5P_DEFAULT)) {
        H5G_stat_t info;

        herr_t status = H5Gget_objinfo(_file_id, pathname.c_str(), 0, &info);
        assert(status == 0);

        result = info.type == H5G_DATASET;
    }

    return result;

}


HDF5Group HDF5File::create_group(
    std::string const& pathname) const
{
    assert(_file_id > 0);
    hid_t group_id = H5Gcreate(_file_id, pathname.c_str(), H5P_DEFAULT,
        H5P_DEFAULT, H5P_DEFAULT);
    assert(group_id > 0);

    return HDF5Group(group_id);
}


void HDF5File::flush()
{
    herr_t status = H5Fflush(_file_id, H5F_SCOPE_GLOBAL);
    assert(status == 0);
}

} // namespace language
} // namespace fern
