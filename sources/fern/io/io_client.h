#pragma once
#include "fern/io/gdal/gdal_client.h"
#include "fern/io/fern/geoneric_client.h"
#include "fern/io/fern/hdf5_client.h"


namespace fern {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class IOClient:
    // Order matters.
    public HDF5Client,
    public GeonericClient,
    public GDALClient
{

public:

protected:

                   IOClient            ();

                   IOClient            (IOClient const&)=delete;

    IOClient&      operator=           (IOClient const&)=delete;

                   IOClient            (IOClient&&)=delete;

    IOClient&      operator=           (IOClient&&)=delete;

    virtual        ~IOClient           ();

private:

};

} // namespace fern
