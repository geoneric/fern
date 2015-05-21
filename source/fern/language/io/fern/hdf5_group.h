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
#include <string>
#include <vector>
#include <hdf5.h>
#include "fern/language/io/fern/hdf5_dataset.h"
#include "fern/language/io/fern/hdf5_type_traits.h"


namespace fern {
namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class HDF5Group
{

public:

                   HDF5Group           ();

    explicit       HDF5Group           (hid_t id);

                   HDF5Group           (HDF5Group const&)=delete;

    HDF5Group&     operator=           (HDF5Group const&)=delete;

                   HDF5Group           (HDF5Group&& other);

    HDF5Group&     operator=           (HDF5Group&& other);

                   ~HDF5Group          ();

    hid_t          id                  () const;

    hsize_t        nr_objects          () const;

    std::string    object_name         (hsize_t index) const;

    hsize_t        nr_groups           () const;

    std::vector<std::string>
                   group_names         () const;

    hsize_t        nr_datasets         () const;

    HDF5Dataset    open_dataset        (std::string const& path);

    template<
        typename T>
    HDF5Dataset    create_dataset      (std::string const& name);

private:

    hid_t          _group_id;

};


template<
    typename T>
inline HDF5Dataset HDF5Group::create_dataset(
    std::string const& name)
{
    assert(_group_id > 0);

    hid_t dataspace_id = H5Screate(H5S_SCALAR);
    assert(dataspace_id > 0);

    hid_t dataset_id = H5Dcreate(_group_id, name.c_str(),
        HDF5TypeTraits<T>::data_type, dataspace_id, H5P_DEFAULT, H5P_DEFAULT,
        H5P_DEFAULT);
    assert(dataset_id > 0);

    herr_t status = H5Sclose(dataspace_id);
    assert(status == 0);

    return HDF5Dataset(dataset_id);
}

} // namespace language
} // namespace fern
