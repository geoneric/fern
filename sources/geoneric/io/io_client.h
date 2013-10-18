#pragma once
#include "geoneric/io/gdal/gdal_client.h"
#include "geoneric/io/geoneric/geoneric_client.h"
#include "geoneric/io/geoneric/hdf5_client.h"


namespace geoneric {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class IOClient:
    // Order matters.
    public GDALClient,
    public HDF5Client,
    public GeonericClient
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

} // namespace geoneric
