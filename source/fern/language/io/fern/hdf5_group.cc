// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/io/fern/hdf5_group.h"


namespace fern {
namespace language {

HDF5Group::HDF5Group()

    : _group_id(0)

{
}


HDF5Group::HDF5Group(
    hid_t id)

    : _group_id(id)

{
    assert(_group_id >= 0);
}


HDF5Group::HDF5Group(
    HDF5Group&& other)

    : _group_id(other._group_id)

{
    other._group_id = 0;
}


HDF5Group& HDF5Group::operator=(
    HDF5Group&& other)
{
    if(_group_id > 0) {
        H5Gclose(_group_id);
    }

    _group_id = other._group_id;
    other._group_id = 0;

    return *this;
}


HDF5Group::~HDF5Group()
{
    if(_group_id > 0) {
        H5Gclose(_group_id);
    }
}


hid_t HDF5Group::id() const
{
    assert(_group_id > 0);
    return _group_id;
}


hsize_t HDF5Group::nr_objects() const
{
    assert(_group_id > 0);
    hsize_t nr_objects;
    herr_t status = H5Gget_num_objs(_group_id, &nr_objects);
    assert(status == 0);
    return nr_objects;
}


hsize_t HDF5Group::nr_groups() const
{
    assert(_group_id > 0);
    H5G_obj_t type;
    hsize_t result = 0;

    for(hsize_t i = 0; i < nr_objects(); ++i) {
        type = H5Gget_objtype_by_idx(_group_id, i);
        assert(type >= 0);

        if(type == H5G_GROUP) {
            ++result;
        }
    }

    return result;
}


hsize_t HDF5Group::nr_datasets() const
{
    assert(_group_id > 0);
    H5G_obj_t type;
    hsize_t result = 0;

    for(hsize_t i = 0; i < nr_objects(); ++i) {
        type = H5Gget_objtype_by_idx(_group_id, i);
        assert(type >= 0);

        if(type == H5G_DATASET) {
            ++result;
        }
    }

    return result;
}


std::string HDF5Group::object_name(
    hsize_t index) const
{
    assert(_group_id > 0);
    ssize_t nr_characters = H5Gget_objname_by_idx(_group_id, index, nullptr,
        0);
    assert(nr_characters >= 0);

    std::vector<char> buffer(nr_characters + 1);

    nr_characters = H5Gget_objname_by_idx(_group_id, index, buffer.data(),
        buffer.size());
    assert(static_cast<size_t>(nr_characters) == buffer.size() - 1);

    std::string name(buffer.begin(), buffer.end() - 1);

    return name;
}


std::vector<std::string> HDF5Group::group_names() const
{
    assert(_group_id > 0);
    H5G_obj_t type;
    std::vector<std::string> result;

    for(hsize_t i = 0; i < nr_objects(); ++i) {
        type = H5Gget_objtype_by_idx(_group_id, i);
        assert(type >= 0);

        if(type == H5G_GROUP) {
            result.push_back(object_name(i));
        }
    }

    return result;
}


HDF5Dataset HDF5Group::open_dataset(
    std::string const& path)
{
    assert(_group_id > 0);
    hid_t dataset_id = H5Dopen(_group_id, path.c_str(), H5P_DEFAULT);
    assert(dataset_id > 0);

    return HDF5Dataset(dataset_id);
}

} // namespace language
} // namespace fern
