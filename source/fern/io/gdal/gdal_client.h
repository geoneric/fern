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

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class GDALClient
{

public:

protected:

                   GDALClient          ();

                   GDALClient          (GDALClient const&)=delete;

    GDALClient&    operator=           (GDALClient const&)=delete;

                   GDALClient          (GDALClient&&)=delete;

    GDALClient&    operator=           (GDALClient&&)=delete;

    virtual        ~GDALClient         ();

private:

    //! Number of times an instance is created.
    static size_t  _count;

    void           register_all_drivers();

    void           deregister_all_drivers();

};

} // namespace fern
