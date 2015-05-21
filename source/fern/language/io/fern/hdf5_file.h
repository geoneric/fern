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
#include <hdf5.h>
#include "fern/language/io/fern/hdf5_group.h"


namespace fern {
namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class HDF5File
{

public:

                   HDF5File            ();

    explicit       HDF5File            (hid_t id);

                   HDF5File            (HDF5File const&)=delete;

                   HDF5File            (HDF5File&&)=delete;

                   ~HDF5File           ();

    HDF5File&      operator=           (HDF5File const&)=delete;

    HDF5File&      operator=           (HDF5File&&)=delete;

    hid_t          id                  () const;

    bool           is_group            (std::string const& pathname) const;

    hid_t          open_group          (std::string const& pathname) const;

    HDF5Group      create_group        (std::string const& pathname) const;

    bool           is_dataset          (std::string const& pathname) const;

    hid_t          open_dataset        (std::string const& pathname) const;

    void           flush               ();

private:

    //! Id of file. Will be close upon exit.
    hid_t          _file_id;

};

} // namespace language
} // namespace fern
