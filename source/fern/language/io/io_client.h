// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/language/io/gdal/gdal_client.h"
#include "fern/language/io/fern/fern_client.h"
#include "fern/language/io/fern/hdf5_client.h"


namespace fern {
namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class IOClient:
    // Order matters.
    public HDF5Client,
    public FernClient,
    public GDALClient
{

public:

                   IOClient            ();

                   IOClient            (IOClient const&)=delete;

    IOClient&      operator=           (IOClient const&)=delete;

                   IOClient            (IOClient&&)=delete;

    IOClient&      operator=           (IOClient&&)=delete;

    virtual        ~IOClient           ();

private:

};

} // namespace language
} // namespace fern
