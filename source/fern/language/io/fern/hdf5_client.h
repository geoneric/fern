// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <cstdlib>


namespace fern {
namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class HDF5Client
{

public:

protected:

                   HDF5Client          ();

                   HDF5Client          (HDF5Client const&)=delete;

    HDF5Client&    operator=           (HDF5Client const&)=delete;

                   HDF5Client          (HDF5Client&&)=delete;

    HDF5Client&    operator=           (HDF5Client&&)=delete;

    virtual        ~HDF5Client         ();

private:

    //! Number of times an instance is created.
    static size_t  _count;

};

} // namespace language
} // namespace fern
