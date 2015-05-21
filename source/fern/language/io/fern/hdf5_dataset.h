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
#include <hdf5.h>
#include "fern/language/io/fern/hdf5_type_traits.h"


namespace fern {
namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class HDF5Dataset
{

public:

                   HDF5Dataset         ();

    explicit       HDF5Dataset         (hid_t id);

                   HDF5Dataset         (HDF5Dataset const&)=delete;

    HDF5Dataset&   operator=           (HDF5Dataset const&)=delete;

                   HDF5Dataset         (HDF5Dataset&& other);

    HDF5Dataset&   operator=           (HDF5Dataset&& other);

                   ~HDF5Dataset        ();

    hid_t          id                  () const;

    H5T_class_t    type_class          () const;

    hid_t          type                () const;

    bool           space_is_simple     () const;

    H5S_class_t    extent_type         () const;

    template<
        typename T>
    void           read                (T& value) const;

    template<
        typename T>
    void           write               (T const& value) const;

private:

    hid_t          _dataset_id;

    hid_t          space               () const;

};


template<
    typename T>
inline void HDF5Dataset::read(
    T& value) const
{
    assert(_dataset_id > 0);
    herr_t status = H5Dread(_dataset_id, HDF5TypeTraits<T>::data_type,
        H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
    assert(status == 0);
}


template<
    typename T>
inline void HDF5Dataset::write(
    T const& value) const
{
    assert(_dataset_id > 0);
    herr_t status = H5Dwrite(_dataset_id, HDF5TypeTraits<T>::data_type,
        H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
    assert(status == 0);
}

} // namespace language
} // namespace fern
