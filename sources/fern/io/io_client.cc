#include "fern/io/io_client.h"


namespace fern {

IOClient::IOClient()

    : HDF5Client(),
      GeonericClient(),
      GDALClient()

{
}


IOClient::~IOClient()
{
}

} // namespace fern
